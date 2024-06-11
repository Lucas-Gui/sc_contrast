import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #silence pandas warning about is_sparse

from data_utils import *
from models import *
from contrastive_data import *
from scoring import *

from configargparse import ArgumentParser
import torch 
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from os.path import join
from glob import glob
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json
import sys
from warnings import warn
from dataclasses import dataclass
from contextlib import nullcontext

import pprint
# config : task -> loss, model, dataset

loss_dict : Dict[str, Type[ContrastiveLoss]] = {
    'standard':SiameseLoss,
    'lecun':LeCunContrastiveLoss,
    'cosine': CosineContrastiveLoss
}

@dataclass
class _Config():
    loss_dict:dict
    model_class:Type[Model]
    dataset_class:Type[Dataset]

config_dict:Dict[str, _Config] = { # task -> config mapping
    'siamese':_Config(loss_dict, Siamese, SiameseDataset),
    'classifier':_Config({'standard': ClassifierLoss}, Classifier, ClassifierDataset),
    'batch-supervised': _Config({'standard':BatchContrastiveLoss}, ContrastiveModel, BatchDataset),
    'cycle-classifier': _Config(
        {'standard':DoubleClassifierLoss}, CycleClassifier, CycleClassifierDataset
        ),

}

bag_dataset_dict = {
    'siamese':None,
    'classifier':ClassifierBagDataset,
    'batch-supervised':BatchBagDataset,
}

# Context : to put all arguments to main that will be passed down to core_loop and train_model,
#   in order to limit the size of function calls/definitions

@dataclass 
class Context(): # defaults to None to allow for partial definition 
    device:str = 'cpu'
    run_dir:str = None
    run_name:str = None
    task:str = None
    k_nn:int = 5
    verbosity:int = 3
    index_dir :str = None
    model_file:str = None
    meta_file :str = None

# ctx = Context() # in a jupyter notebook, assign the correct values to this instance #TODO : can we remove ?

def make_dir_if_needed(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        if not os.path.isdir(path):
            raise FileExistsError(path)
        
def get_paths(data_dir:str, subset : Literal['processed','raw', 'filtered'] = 'processed'): # TODO move to data_utils
    '''
    Reads and returns, in that order :
        <gene>.<subset>.matrix.mtx.gz, '''
    if subset == 'processed':
        _r = '.processed'
    elif subset == 'raw':
        _r = '.rawcounts'
    elif subset == 'filtered':
        _r = '.filtered'
    else:
        raise ValueError('subset should be "processed", "raw" or "filtered"')
    paths = []
    for p in [_r+'.matrix.mtx.gz',_r+'.genes.csv.gz',_r+'.cells.csv.gz', '.variants2cell.csv.gz', 
              '.variants.csv', '.cells.metadata.csv.gz']:
        l = glob(join(data_dir, '*'+p)) #note : this means that variants2cell, variants and cells.metadata can all have any prefix
        assert len(l)==1, f"There should be exaclty one match for {join(data_dir, '*'+p)}, {len(l)} found."
        paths.extend(l)
    return paths

def get_counts(args):
    '''Load data from args.data_path, and filter if args.filter_variants is not None.'''
    paths = get_paths(args.data_path, subset=args.data_subset)
    print(f'Loading data from {args.data_path}...', flush=True)
    filt = None
    if args.filter_variants is not None:
        filt = pd.read_csv(args.filter_variants, header=None).squeeze()
        filt = filt.str.upper()
        print(f'Filtering for {filt.values}')
    counts = load_data(*paths, group_wt_like= args.group_synon, filt_variants=filt,
                       standardize=args.data_subset != 'processed')
    return counts

def load_split_df(index_dir, counts:pd.DataFrame, reorder_categories = True,
               ) -> List[pd.DataFrame]:
    '''
    If reorder_categories, use saved category order and drop unused categories. This is important for classifier models,
    as category order defines the label.
    Set it to false if the categories are not the same as during training ("control" split into individual variants, for example)
    '''
    i = 0
    dataframes = []          
    if reorder_categories:
        cat = pd.read_csv(join(index_dir, 'categories.csv'), index_col=0).to_numpy().flat
        counts = counts[counts['variant'].isin(cat)].copy() # drop unused variants
        counts['variant'] = counts.variant.cat.remove_unused_categories().cat.reorder_categories(cat)
    else:
        warn('Not reordering categories will lead to wrong classifier prediction')
    while os.path.isfile(join(index_dir, f'index_{i}.csv')):
        idx = pd.read_csv(join(index_dir, f'index_{i}.csv'), index_col=1).index
        dataframes.append(counts.loc[idx])
        i+=1
    print(', '.join([str(df.shape[0]) for df in dataframes]) + ' exemples in data')
    print(f"{dataframes[0].variant.nunique()} variants in train")
    print(f"{dataframes[2].variant.nunique()} variants in unseen")

    return dataframes

def load_split(index_dir, data:Data, reorder_categories = True,
               ) -> List[Data]:
    '''
    If reorder_categories, use saved category order and drop unused categories. This is important for classifier models,
    as category order defines the label.
    Set it to false if the categories are not the same as during training ("control" split into individual variants, for example)
    '''
    i = 0
    containers = []          
    if reorder_categories:
        cat = pd.read_csv(join(index_dir, 'categories.csv'), index_col=0).to_numpy().flat
        data = data.subset(data.variants.isin(cat)) # drop unused variants
        data.variants = data.variants.cat.remove_unused_categories().cat.reorder_categories(cat)
    else:
        warn('Not reordering categories will lead to wrong classifier prediction')
    parent_idx = data.variants.index
    while os.path.isfile(join(index_dir, f'index_{i}.csv')):
        idx = pd.read_csv(join(index_dir, f'index_{i}.csv'), index_col=1).index
        containers.append(data.subset(parent_idx.isin(idx)))
        i+=1
    print(', '.join([str(len(data)) for data in containers]) + ' exemples in data')
    print(f"{containers[0].variants.nunique()} variants in train")
    print(f"{containers[2].variants.nunique()} variants in unseen")

    return containers

def write_metrics(metrics:  Dict[str, float|dict], writer:SummaryWriter, main_tag:str, i):
    for tag, metric_or_dict in metrics.items():
        if isinstance(metric_or_dict, dict):
            writer.add_scalars(f'{main_tag}/{tag}', metric_or_dict, i)
        else:
            if not tag.startswith('_'):
                writer.add_scalar(f'{main_tag}/{tag}', metric_or_dict, i)


def train_model(train, test_seen, test_unseen, model, run_meta, 
                loss_fn, optimizer, scheduler:optim.lr_scheduler.LRScheduler, writer:SummaryWriter,
                ctx:Context, n_epoch=10_000, 
                 ):
    i_0 = run_meta['i']
    stop_score = f'{ctx.k_nn}_nn_ref' # TODO : CHANGE TO SELF OR SET IN ARGS/CONTEXT
    print(optimizer)
    print('Early stopping metric : ', stop_score)
    bar = tqdm(range(i_0, i_0+n_epoch), position=0, disable= ctx.verbosity <=1)
    best_score = - np.inf
    for i in bar:
        bar.set_postfix({'i':i})
        metrics_train = core_loop(train, model, loss_fn, ctx, optimizer, mode='train')
        write_metrics(metrics_train, writer, 'train', i)
        
        metrics_seen =  core_loop(test_seen, model, loss_fn, optimizer=None, mode='test', ctx=ctx)
        metrics_seen[f'{ctx.k_nn}_nn_ref'] = knn_ref_score(
            model, train, 
            test_seen, k=ctx.k_nn, device=ctx.device)
        write_metrics(metrics_seen, writer, 'test_seen',i)
        if metrics_seen[stop_score] > best_score:
            best_score = metrics_seen[stop_score]
            torch.save(model, join(ctx.run_dir, 'best_model.pkl'))
            with open(join(ctx.run_dir, 'best_score.json'), 'w') as file:
                json.dump(
                    {'i':i, f'{stop_score}_seen':best_score, }, 
                    file, sort_keys=True, indent=2)
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metrics_seen[f'{ctx.k_nn}_nn_ref'])
        elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step(i)

        if test_unseen is not None:
            metrics_unseen = core_loop(test_unseen, model, loss_fn, ctx=ctx, optimizer=None, mode='test', unseen=True)
            write_metrics(metrics_unseen, writer, 'test_unseen',i)
        #saving and writing
        torch.save(model, ctx.model_file)
        run_meta['i'] = i
        with open(ctx.meta_file, 'w') as file:
            json.dump(run_meta, file, sort_keys=True, indent=2)
    writer.flush()



def core_loop(data:DataLoader, model:Model, loss_fn:ContrastiveLoss, ctx:Context,
              optimizer:torch.optim.Optimizer=None, mode:Literal['train','test']='test',
              unseen=False)-> Tensor:
    if mode == 'train':
        model.train()
    elif mode == 'test':
        model.eval()

    loss_l = []
    y_l = []
    d_l = []
    norm_l = []
    embeds = [] # store embeds for scoring
    labels = [] # store labels for scoring
    with torch.no_grad() if mode=='test' else nullcontext(): # disable grad only in test mode
        for x,y, in tqdm(data, position=1, desc=f'{mode}ing loop', leave=False, disable= ctx.verbosity <=1):
            x = [x_i.to(ctx.device) for x_i in x]
            y = [y_i.to(ctx.device) for y_i in y]
            outputs, emb = model.forward(*x)
            if not unseen  or ctx.task not in ['classifier','cycle-classifier']: #avoid trying to classfify unseen classes
                loss:Tensor = loss_fn.forward(*outputs, *y)
                loss_l.append(loss)
            if mode == 'train':
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # keep first embed and label
            embeds.append(emb)
            labels.append(y[0])
            if ctx.task == 'siamese':
                y_l.append(y[0] == y[1])
                e1, e2,  = outputs
                d_l.append(torch.norm(e1-e2, p=2, dim=-1))
                norm_l.append((torch.norm(e1, dim=-1) + torch.norm(e2, dim=-1))*loss_fn.alpha)
    labels = torch.concat(labels)
    embeds = torch.concat(embeds)
 
    metrics = {
        f'{ctx.k_nn}_nn_self' : knn_self_score(embeds, labels)
    }
    if not unseen  or ctx.task not in ['classifier', 'cycle-classifier']: 
        metrics['loss'] = np.mean( [t.detach().cpu().item() for t in loss_l])
    if mode == 'train':
        metrics['lr'] = optimizer.param_groups[0]['lr']
    if ctx.task in ['siamese']:
        y = torch.concat(y_l).detach().cpu()
        d = torch.concat(d_l).detach().cpu()
        metrics.update({
        'dist_pos' : ((d*y).sum()/y.sum()).item(),
        'dist_neg' : ((d* ~y).sum()/(~y).sum()).item(),
        'l2_penalty': torch.concat(norm_l).cpu().mean().item(),
        'roc': ROC_score(y, d)[0].item(),
        })
    return metrics

def split_data(data:Data, ctx:Context, restart, load_split_path=None, unseen_frac=0.25, cell_frac=0.25 ):
    '''
    Split the data into train, test_seen and test_unseen.
    If restart, load split from index dir
    If load_split_path is not None, load split from that directory
    Otherwise, create a new split.
    In all cases, save the indices to ctx.index_dir
    '''
    if restart : #load model and split
        containers = load_split(ctx.index_dir, data)     
    else:
        if load_split_path is not None:
            print(f'Copying split from {load_split_path}')
            containers = load_split(load_split_path, data)
        else:
            print('Creating new data split')
            containers = split(data, x_var=unseen_frac, x_cell=cell_frac) #random split
        make_dir_if_needed(ctx.index_dir)
        pd.DataFrame(containers[0].variants.cat.categories).to_csv(join(ctx.index_dir, 'categories.csv')) #save category order
    for i, d in enumerate(containers):
        if d is not None:
            df = pd.DataFrame(index=d.variants.index).reset_index()
        else:
            df = pd.DataFrame(columns=['index'])
        df.to_csv(join(ctx.index_dir,f'index_{i}.csv'))
    return containers

def main(args, data:Data, ctx:Context):
    print(f"{ctx.run_dir=}")
    # split data
    data_containers = split_data(
        data, ctx, args.restart, args.load_split, args.unseen_frac)
    for data in data_containers:
        if data is not None:
            data.compute_y() # freeze y once variants are set
    # get loss/model/dataset classes for task
    config = config_dict[args.task]
    if args.bag_size >= 1:
        try : 
            config.dataset_class = bag_dataset_dict[args.task]
        except KeyError:
            raise NotImplementedError(f'Bagging is not implemented for task {args.task}')
    if args.restart : #load model and split
        print(f'Loading model from {ctx.model_file}')
        model = torch.load(ctx.model_file)
        with open(ctx.meta_file, 'r') as file:
            run_meta = json.load(file) # data that we want to keep between restarts
    else:
        # make new model
        n_class = data_containers[0].variants.nunique()
        in_shape = data_containers[0].x.shape[1]
        print(f'{n_class} classes, {in_shape} features\n', flush=True)
        inner_network = MLP(input_shape=in_shape, inner_shape=args.shape, dropout=args.dropout,
                    output_shape=args.embed_dim, normalize= not args.no_norm_embeds,)
        if args.bag_size >= 1:
            match args.mil_mode:
                case 'attention':
                    inner_network = AttentionMIL(inner_network, inner_shape=64)
                case 'mean':
                    inner_network = AverageMIL(inner_network, )
                case _:
                    raise NotImplementedError(f'MIL mode {args.mil_mode} is not implemented')                
        model = config.model_class(
               inner_network, 
            #task-specific kwargs
            n_class = n_class, # should be equal to nb of codes 
            projection_shape=args.projection_shape, # shape of projection network for contrastive models. None for no projection
                            ).to(ctx.device)
        run_meta = {
            'i':0,
        }
    run_meta['n_variants'] = data_containers[0].variants.nunique()
    train, test_seen, test_unseen = make_loaders(
        *data_containers, batch_size=args.batch_size, n_workers=args.n_workers, 
         dataset_class=config.dataset_class, dataset_kwargs={'bag_size':args.bag_size},
        device=ctx.device)
    in_shape = next(iter(train))[0][0].shape[1]
    print(model)
    loss_fn = config.loss_dict[args.loss](margin=args.margin, alpha=args.alpha)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=args.patience)
    elif args.scheduler == 'restarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.cosine_t, 
        )
    else :
        raise ValueError('Please specify a valid scheduler')
    # models/exp/run -> runs/exp/run
    writer = SummaryWriter(join('runs',*(ctx.run_dir.split('/')[1:])))

    train_model(train, test_seen, test_unseen, model, run_meta,
                loss_fn, optimizer=optimizer, scheduler=scheduler, writer=writer,
                n_epoch=args.n_epochs, ctx=ctx
                )
    
def test(args): #TODO : Fix. in particular, make a fake data directory with all the necessary files to use load_data
    print('---RUNNING TEST--- ')
    args.data_path = None
    args.restart = False
    make_dir_if_needed(_run_dir)
    counts = pd.read_csv('/home/lguirardel/data/perturb_comp/data/KRAS_test.csv', index_col=0)

    main(args, counts, unseen_frac=0, ctx=Context())


def make_parser():
    parser = ArgumentParser('''Train and evaluate a contrastive model on Ursu et al. data''')
    parser.add_argument('data_path', help='Path to data directory. ')
    parser.add_argument('run_name',)

    parser.add_argument('--dest-name', default='', help='Optionally store model and runs results in subdir')
    parser.add_argument('--verbose', default=2, help='Verbosity level. Set to 1 to silence tqdm output', type=int)

    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--load-split',metavar='RUN', help='If passed, load split fron given path. Use to compare models on the same data', default=None)
    parser.add_argument('--data-subset', default='processed', choices=['processed','raw','filtered'], help='Data version to use')
    parser.add_argument('--unseen-frac', default=0.25, type=float, help='Fraction of unseen variants')
    
    parser.add_argument('--loss', choices=[*loss_dict.keys()], default='standard',
                        help='''standard loss : $y ||e_1 - e_2||^2_2 + (1-y) max(||e_1 - e_2||_2 -m, 0)^2 $''')
    parser.add_argument('-m','--margin',default=1, type=float, help='Contrastive loss margin parameter')
    parser.add_argument('-a','--alpha',default =0, type=float, help='L2 embedding regularization')
    parser.add_argument('-d','--dropout',default=0.2, type=float, help='Dropout frequency')
    parser.add_argument('-w','--weight-decay',default=1e-2, type=float, help='Weight decay parameter')
    parser.add_argument('--projection-shape', type=int, nargs='+', help='Projection MLP shape', default=None)
    parser.add_argument('--batch-size',default=128, help='Batch size', type=int)
    parser.add_argument('--positive-fraction',default=0.5, help='Fraction of positive training samples', type=float)
    parser.add_argument('--shape', type=int, nargs='+', help = 'MLP shape', default=[100, 100])
    parser.add_argument('--embed-dim', type=int, default=20 ,help='Embedding dimension') # Ursu et al first project in 50 dim, but only use the 20 first ones for sc-eVIP
    
    parser.add_argument('-n','--n-epochs', metavar='N', default=600, type=int, help='Number of epochs to run')
    parser.add_argument('--group-synon',action='store_true', 
                        help='If passed, group all synonymous variants in the same class')
    parser.add_argument('--filter-variants', metavar='FILE', help='Path to file with variants to include. If not passed, all variants are included', default = None)
    # parser.add_argument('--subsample-variants', metavar='p', type=float, help='Subsample p fraction of variants', default=None)
    parser.add_argument('--no-norm-embeds',action='store_true',
                        help='If passed, do not rescale emebeddings to unit norm')
    # scheduler lr args
    sched_args = parser.add_argument_group('Learning rate scheduler arguments')
    sched_args.add_argument('--lr',type=float, default=1e-3, )
    sched_args.add_argument('--scheduler', choices=['plateau','restarts'], default='plateau',
                            help="plateau : reduce lr on plateau\nrestarts : cosine annealaing with warm restarts")
    sched_args.add_argument('--patience',type=int,help='Patience for reduce lr on plateau', default=40)
    sched_args.add_argument('--cosine-t',type=int,help='Period for cosine annealing', default=100)

    parser.add_argument('--task',choices=[*config_dict.keys()], help='Type of learning task to optimize', default='classifier')
    parser.add_argument('--knn', default=5, type=int, help='Number of neighbors for knn scoring')
    parser.add_argument('--bag-size', '--instance', default=0, type=int, help='Number of bag_sizes to use for Multiple Instance Learning. If 0, do not use MIL')
    parser.add_argument('--mil-mode', choices=['attention','mean'], default='attention', help='MIL aggregation mode')

    parser.add_argument('-c', '--config-file', is_config_file_arg=True, help='add config file')
    parser.add_argument('--n-workers', default=0, type=int, help='Number of workers for datalaoding')
    parser.add_argument('--overwrite', action='store_true', help='Do not prompt for confirmation if model already exists')
    parser.add_argument('--run-test', help='Run test on reduced data', action='store_true')
    parser.add_argument('--cpu', help='Run on cpu even if gpu is available', action='store_true')
    return parser

IGNORE_ARGS_CONFIG = ['restart']
def file_config_ignore(args, sources):
    '''
    Update args to ignore the values of some arguments when passed from a file
    '''
    # ignore some arguments if they come from a file
    # iiuc, configargparse will store key in _source_to_setting.config ONLY if it not superseded by a CL arg
    for arg in IGNORE_ARGS_CONFIG:
        for key in sources.keys():
            if key.startswith('config_file'):
                if sources[key].get(arg) is not None :
                    value = sources[key][arg][0].default
                    print(f'Ignoring argument {arg} with value {args.__dict__[arg]} from {key}, setting to default value {value}')
                    args.__dict__[arg] = value

def args_check(args, sources):
    '''argument compatibility check. Raise errors or print warnings if some arguments are incompatible.'''
    if args.restart:
        for arg in ['shape','embed_dim']:
            if sources['command_line'].get(arg) is not None:
                warn(f'Warning : restart : {arg} will be ignored', )
        if args.load_split is not None:
            warn('Argument load_split will be ignored in favor of split saved for this model previous instance')
    if (not args.no_norm_embeds and args.alpha ) and not args.task == 'cycle-classifier': # for cycle classifier, alpha is the cycle/variant weight
        warn('Embedding norm penalty parameter alpha is nonzero while embeddings are normalized.')
    if args.task in ['classifier'] :
        if args.projection_shape is not None:
            print('Warning : projection shape is ignored for classifier task')
        if args.alpha : 
            raise NotImplementedError('Nonzero embedding norm penalty parameter alpha is not compatible with a classification task.')
      
if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    _run_dir = join('models', args.dest_name, args.run_name) if not args.run_test else join('models', '_test')
    # run test
    if args.run_test:
        test(args)
        print('Test concluded.')
        sys.exit()
    
    # args check and config file filtering
    sources = parser._source_to_settings
    file_config_ignore(args, sources)
    args_check(args, sources)

    # safety overwriting check
    if os.path.exists(join(_run_dir, 'config.ini')) and not args.restart and not args.overwrite:
        check = input(f'{_run_dir} already exists, are you sure you want to overwrite ? [Y/n] ')
        if check.lower() in ['n','no']:
            print('Exiting.')
            sys.exit()
        else:
            print('Proceeding.')
    if args.dest_name:
        make_dir_if_needed(join('models', args.dest_name))
        make_dir_if_needed(join('runs', args.dest_name))
    make_dir_if_needed(_run_dir)
    # save args to file
    parser.write_config_file(args, [join(_run_dir, 'config.ini')], exit_after=False)
    # print(parser._source_to_settings)
    # load variants to include

    print()
    _device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu' 
    print(f'Using {_device}.')
    _counts = get_counts(args)
    ctx = Context(
        _device, _run_dir, args.run_name, task=args.task, k_nn=args.knn, 
        verbosity=args.verbose,
        index_dir = join(_run_dir, 'split'),
        model_file = join(_run_dir,'model.pkl'),
        meta_file = join(_run_dir, 'meta.json'),
    )
    data = Data.from_df(_counts, device=_device)
    main(args, data, ctx=ctx, )
    
