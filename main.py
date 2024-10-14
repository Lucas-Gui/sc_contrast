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
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json
import sys
from warnings import warn
from dataclasses import dataclass
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
# config : task -> loss, model, dataset

SAVE_FREQUENCY = 100

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

class DataSource(Enum):
    BHUIYAN = 1
    URSU = 2
    COOPER = 3

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
    data_source: DataSource = None

# ctx = Context() # in a jupyter notebook, assign the correct values to this instance #TODO : can we remove ?

class EarlyStoppingIterator():
    '''
    If loss does not improve for n iterations, stop training.
    Smooth loss by rolling average.
    '''
    def __init__(self, dt, roll, max_iter=np.inf, i0 =0):
        self.max_iter = max_iter
        self.iter = i0
        self.dt = dt
        self.roll = roll
        self.losses = np.ones(dt+roll)

    def __iter__(self):
        return self
    
    def update(self, loss):
        self.losses[:-1] = self.losses[1:]
        self.losses[-1] = loss

    def __next__(self):
        if self.iter >= self.max_iter:
            raise StopIteration
        if self.iter >= self.dt+self.roll:
            current = np.mean(self.losses[-self.roll:])
            ref = np.mean(self.losses[:self.roll])
            if current >= ref:
                raise StopIteration
        self.iter += 1
        return self.iter
    
class RangeIterator():
    '''
    Wrapper around range for compatibility with EarlyStoppingIterator
    '''
    def __init__(self, *args):
        self.args = args
    
    def __iter__(self):
        return range(*self.args).__iter__()
    
    def update(self, loss):
        pass

def make_dir_if_needed(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        if not os.path.isdir(path):
            raise FileExistsError(path)


def get_counts(args):
    '''Load data from args.data_path, and filter if args.filter_variants is not None.'''
    filt = None
    if args.filter_variants is not None:
        filt = pd.read_csv(args.filter_variants, header=None).squeeze()
        filt = filt.str.upper()
        print(f'Filtering for {filt.values}')
    print(f'Loading data from {args.data_path}...', flush=True)
    if args.cooper:
        counts = load_Cooper_data(
            args.data_path, group_wt_like=args.group_synon, filt_variants=filt,
            standardize=True, filt_cells=args.filter_cells,log1p=args.log1p, n_cell_min=args.cell_min
            )
    else:
        paths = get_paths(args.data_path, subset=args.data_subset)
        counts = load_data(*paths, group_wt_like= args.group_synon, filt_variants=filt,
                       standardize=args.data_subset != 'processed', filt_cells=args.filter_cells,
                       log1p=args.log1p, n_cell_min=args.cell_min)
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


def write_metrics(metrics:  Dict[str, float|dict], writer:SummaryWriter, main_tag:str, i):
    for tag, metric_or_dict in metrics.items():
        if isinstance(metric_or_dict, dict):
            writer.add_scalars(f'{main_tag}/{tag}', metric_or_dict, i)
        else:
            if not tag.startswith('_'):
                writer.add_scalar(f'{main_tag}/{tag}', metric_or_dict, i)


def train_model(train, test_seen, test_unseen, model, run_meta, 
                loss_fn, optimizer, scheduler:optim.lr_scheduler.LRScheduler, writer:SummaryWriter,
                ctx:Context, n_epoch=600, early_stop=None,
                 ):
    assert early_stop is None or n_epoch is None, 'Only one of n_epoch and early_stop can be passed'
    i_0 = run_meta['i']
    stop_score = f'{ctx.k_nn}_nn_ref' # TODO :  SET IN ARGS/CONTEXT
    print(optimizer)
    print('Early stopping metric : ', stop_score)
    if early_stop is not None:
        iterator = EarlyStoppingIterator(early_stop, 20, i0=i_0) # TODO : expose rolling window size in args
    else:
        iterator = RangeIterator(i_0, i_0+n_epoch)
    bar = tqdm(iterator, position=0, disable= ctx.verbosity <=2)
    best_score = - np.inf
    best_i = last_save_model = i_0
    for i in bar:
        bar.set_postfix({'i':i})
        metrics_train = core_loop(train, model, loss_fn, ctx, optimizer, mode='train')
        write_metrics(metrics_train, writer, 'train', i)
        
        metrics_seen =  core_loop(test_seen, model, loss_fn, optimizer=None, mode='test', ctx=ctx)
        metrics_seen[f'{ctx.k_nn}_nn_ref'] = knn_ref_score(
            model, train, 
            test_seen, k=ctx.k_nn, device=ctx.device)
        write_metrics(metrics_seen, writer, 'test_seen',i)
        iterator.update(- metrics_seen[stop_score])
        if metrics_seen[stop_score] > best_score:
            best_model = deepcopy(model)
            best_score = metrics_seen[stop_score]
            best_i = i
        if i % SAVE_FREQUENCY == 0 and best_i > last_save_model: # save model every N epochs, IF it has improved
            last_save_model = i
            torch.save(best_model, join(ctx.run_dir, 'best_model.pkl'))
            with open(join(ctx.run_dir, 'best_score.json'), 'w') as file:
                json.dump(
                    {'i':best_i, f'{stop_score}_seen':best_score, }, 
                    file, sort_keys=True, indent=2)
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metrics_seen[f'{ctx.k_nn}_nn_ref'])
        elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step(i)

        if test_unseen is not None:
            metrics_unseen = core_loop(test_unseen, model, loss_fn, ctx=ctx, optimizer=None, mode='test', unseen=True)
            write_metrics(metrics_unseen, writer, 'test_unseen',i)
        #saving and writing
        if i % SAVE_FREQUENCY == 0:
            torch.save(model, ctx.model_file)
            run_meta['i'] = i
            with open(ctx.meta_file, 'w') as file:
                json.dump(run_meta, file, sort_keys=True, indent=2)


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
        for x,y, in tqdm(data, position=1, desc=f'{mode}ing loop', leave=False, disable= ctx.verbosity <=2):
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

def split_data(data:Data, ctx:Context, restart, 
               load_split_path=None, unseen_frac=None, cell_frac=0.25, k_var=None ,
               verbosity=3 ) -> List[List[SlicedData]]:
    '''
    Split the data into folds of train, test_seen and test_unseen.
    If restart, load split from index dir.
    If load_split_path is not None, load split from that directory
    Otherwise, create a new split (or raise an exception if that split exists). 
    In all cases, save the indices to ctx.index_dir
    '''
    print('Splitting data...', )
    t = datetime.now()
    if restart : #load model and split
        containers = load_split(ctx.index_dir, data)     
    else:
        if load_split_path is not None:
            if verbosity > 1:
                print(f'Copying split from {load_split_path}')
            containers = load_split(load_split_path, data)
        else:
            assert (k_var is None) ^ (unseen_frac is None), 'If not loading existing split, exactly one of k_var or unseen_frac must be passed'
            if os.path.exists(ctx.index_dir):
                raise FileExistsError(f'Index directory {ctx.index_dir} already exists.')
            if verbosity > 1:
                print('Creating new data split')
            make_dir_if_needed(ctx.index_dir)
            if k_var is not None:
                containers = var_k_fold_split(data, k=k_var, x_cell=cell_frac, save_dir=ctx.index_dir) 
            else:
                containers = [split(data, x_cell=cell_frac, x_var=unseen_frac, save_dir=ctx.index_dir)]
    print(f'Split done in {(datetime.now()-t).total_seconds():.1f} s')
    return containers

def main(args, data:Data|List[SlicedData], ctx:Context):
    '''
    If data is a Data object, will split it according to args and ctx. This creates copies and should be avoided if possible when training multiple models.
    Otherwise, data is assumed to be split as a list of SlicedData objects.
    '''
    if ctx.verbosity > 0:
        print(f"{ctx.run_dir=}")
    # split data iff it is not already split
    if isinstance(data, Data):
        folds = split_data(
            data, ctx, args.restart, args.load_split, args.unseen_frac)
        data_containers = folds[args.fold] if args.fold is not None else folds[0]
    else:
        if ctx.verbosity > 1:
            print('Data is already split')
        data_containers = data
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
        in_shape = data_containers[0][0].x.shape[0]
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
         subsampling_t=args.subsample, 
         verbosity=ctx.verbosity,device=ctx.device)
    in_shape = next(iter(train))[0][0].shape[1]
    print(model)
    loss_fn = config.loss_dict[args.loss](margin=args.margin, alpha=args.alpha)
    print(type(loss_fn), flush=True)
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
    runs_dir = join('runs',*(ctx.run_dir.split('/')[1:]))
    if os.path.exists(runs_dir):
        if args.restart:
            if ctx.verbosity > 0:
                print(f'Previous run in {runs_dir} exists, not cleaning up because restart option has been passed')
        else:
            if ctx.verbosity > 0:
                print(f'Cleaning up previous runs in {runs_dir}')
            for f in os.listdir(runs_dir):
                os.remove(join(runs_dir, f))
    writer = SummaryWriter(runs_dir, max_queue=10)

    train_model(train, test_seen, test_unseen, model, run_meta,
                loss_fn, optimizer=optimizer, scheduler=scheduler, writer=writer,
                n_epoch=args.n_epochs, early_stop=args.early_stop, ctx=ctx
                )
    
def test(args): #TODO : Fix. in particular, make a fake data directory with all the necessary files to use load_data
    print('---RUNNING TEST--- ')
    args.data_path = None
    args.restart = False
    make_dir_if_needed(_run_dir)
    counts = pd.read_csv('/home/lguirardel/data/perturb_comp/data/KRAS_test.csv', index_col=0)

    main(args, counts, unseen_frac=0, ctx=Context())


def make_parser():
    parser = ArgumentParser(description='''Train and evaluate a contrastive model on Ursu et al. data''')
    parser.add_argument('data_path', help='Path to data directory. ')
    parser.add_argument('run_name',)

    parser.add_argument('--cooper', action='store_true', help='Use Cooper et al. data')

    parser.add_argument('--dest-name', default='', help='Optionally store model and runs results in subdir')
    parser.add_argument('--verbose', default=2, help='Verbosity level. Set to 1 to silence tqdm output', type=int)

    epochs_args = parser.add_mutually_exclusive_group(required=False) # one of the two should be passed, but the options to pass neither is left to allow for empty args creation
    epochs_args.add_argument('-n','--n-epochs', metavar='N', default=None, type=int, help='Number of epochs to run',) 
    epochs_args.add_argument('--early-stop', metavar='N', type=int, help='Stop after N epochs without improvement', default=None)

    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--load-split',metavar='RUN', help='If passed, load split fron given path. Use to compare models on the same data', default=None)
    parser.add_argument('--data-subset', default='processed', choices=['processed','raw','filtered'], help='Data version to use')
    parser.add_argument('--unseen-frac', default=0.25, type=float, help='Fraction of unseen variants')
    parser.add_argument('--fold', default=None, type=int, help='If loading a k-fold split, specify the fold to use')

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
    parser.add_argument('--no-norm-embeds',action='store_true',
                        help='If passed, do not rescale emebeddings to unit norm')
    parser.add_argument('--group-synon',action='store_true', 
                        help='If passed, group all synonymous variants in the same class')
    # data preprocessing args
    parser.add_argument('--filter-cells', action='store_true', help='If passed, filter cells based on counts, number of expressed genes, and mitochondrial counts.')
    parser.add_argument('--filter-variants', metavar='FILE', help='Path to file with variants to include. If not passed, all variants are included', default = None)
    parser.add_argument('--log1p', action='store_true', help='If passed, log1p transform the data')
    parser.add_argument('--subsample', default=None, type=float,
                        help='''Subsample variants. If a float in (0,1), set the subsampling threshold to the corresponding quantile of the cell-per-variant counts.
                        If an int, set the number of cells to keep. 
                        ''')
    parser.add_argument('--cell-min', metavar='N', type=int, help='Minimum number of cells under which variants are discarded', default=10)
    # parser.add_argument('--subsample-variants', metavar='p', type=float, help='Subsample p fraction of variants', default=None)

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
    if args.cooper:
        if args.subset != 'raw':
            print('Warning: Cooper data is only available in raw form. Ignoring subset argument')
        if args.group_synon:
            raise NotImplementedError('Grouping synonymous variants is not implemented for Cooper data')
    if args.subsample is not None and args.bag_size > 0:
        raise NotImplementedError('Subsampling is not implemented for MIL models')
    assert args.fold is None or args.fold > 0, "Fold 0 is reserved for control variants"
        
        
      
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
    
