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


# config 

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

config_dict:Dict[str, _Config] = {
    'siamese':_Config(loss_dict, Siamese, SiameseDataset),
    'classifier':_Config({'standard': ClassifierLoss}, Classifier, ClassifierDataset),
    'batch-supervised': _Config({'standard':BatchContrastiveLoss}, Siamese, BatchClassDataset),
    'cycle-classifier': _Config(
        {'standard':DoubleClassifierLoss}, CycleClassifier, CycleClassifierDataset
        )

}

# context -> to put all global variables in the same place

@dataclass 
class Context():
    device:str = 'cpu'
    run_dir:str = None
    run_name:str = None
    task:str = None
    k_nn:int = 3

ctx = Context() # in a jupyter notebook, assign the correct values to this instance

def make_dir_if_needed(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        if not os.path.isdir(path):
            raise FileExistsError(path)
    

def get_paths(data_dir:str):
    '''
    Reads and returns, in that order :
        <variant>.processed.matrix.mtx.gz, '''
    _r = '.processed'
    paths = []
    for p in [_r+'.matrix.mtx.gz',_r+'.genes.csv.gz',_r+'.cells.csv.gz', '.variants2cell.csv.gz', 
              '.variants.csv', '.cells.metadata.csv.gz']:
        l = glob(join(data_dir, '*'+p))
        assert len(l)==1, f"There should be exaclty one match for {join(data_dir, '*'+p)}"
        paths.extend(l)
    return paths

def load_split(index_dir, counts:pd.DataFrame, reorder_categories = True) -> List[pd.DataFrame]:
        '''
        If reorder_categories, use saved category order. This is important for classifier models,
        as category order defines the label.
        Set it to false if the categories are not the same as during training ("control" split into individual variants, for example)
        '''
        i = 0
        dataframes = []
        if reorder_categories:
            cat = pd.read_csv(join(index_dir, 'categories.csv'), index_col=0).to_numpy().flat
            counts['variant'] = counts.variant.cat.reorder_categories(cat)
        else:
            warn('Not reordering categories will lead to wrong classifier prediction')
        while os.path.isfile(join(index_dir, f'index_{i}.csv')):
            idx = pd.read_csv(join(index_dir, f'index_{i}.csv'), index_col=1).index
            dataframes.append(counts.loc[idx])
            i+=1
        
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


        
def train_model(train, test_seen, test_unseen, model, run_meta, model_file, meta_file,
                loss_fn, optimizer, scheduler:optim.lr_scheduler.LRScheduler,
                n_epoch=10_000, 
                 ):
    i_0 = run_meta['i']
    stop_score = f'{ctx.k_nn}_nn_ref'
    print(optimizer)
    bar = tqdm(range(i_0, i_0+n_epoch), position=0)
    writer = SummaryWriter(join('runs',run_name))
    best_score = - np.inf
    for i in bar:
        bar.set_postfix({'i':i})
        metrics_train = core_loop(train, model, loss_fn, optimizer, mode='train')
        write_metrics(metrics_train, writer, 'train', i)
        
        metrics_seen =  core_loop(test_seen, model, loss_fn, optimizer=None, mode='test')
        metrics_seen[f'{ctx.k_nn}_nn_ref'] = knn_ref_score(model, train.dataset.x,test_seen.dataset.x, train.dataset.y, test_seen.dataset.y, k=1, device=ctx.device)
        write_metrics(metrics_seen, writer, 'test_seen',i)
        if metrics_seen[stop_score] > best_score:
            best_score = metrics_seen[stop_score]
            torch.save(model, join(run_dir, 'best_model.pkl'))
            with open(join(run_dir, 'best_score.json'), 'w') as file:
                json.dump({'i':i, f'{stop_score}_seen':best_score}, file, sort_keys=True, indent=2)
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metrics_seen[f'{ctx.k_nn}_nn_ref'])
        elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step(i)

        if test_unseen is not None:
            metrics_unseen = core_loop(test_unseen, model, loss_fn, optimizer=None, mode='test', unseen=True)
            write_metrics(metrics_unseen, writer, 'test_unseen',i)
        #saving and writing
        torch.save(model, model_file)
        run_meta['i'] = i
        with open(meta_file, 'w') as file:
            json.dump(run_meta, file, sort_keys=True, indent=2)
    writer.flush()



def core_loop(data:DataLoader, model:Model, loss_fn:ContrastiveLoss, 
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
        for x,y, in tqdm(data, position=1, desc=f'{mode}ing loop', leave=False, ):
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


def main(args, counts, unseen_frac = 0.25):
    print(f"{run_dir=}")

    index_dir = join(run_dir, 'split')
    model_file = join(run_dir,'model.pkl')
    meta_file = join(run_dir, 'meta.json')

    config = config_dict[args.task]
     
    if args.restart : #load model and split
        dataframes = load_split(index_dir, counts)
        print(f'Loading model from {model_file}')
        model = torch.load(model_file)
        with open(meta_file, 'r') as file:
            run_meta = json.load(file) # data that we want to keep between restarts
    else:
        if args.load_split is not None:
            print(f'Copying split from {args.load_split}')
            dataframes = load_split(join('models',args.load_split,'split'), counts)
        else:
            dataframes = split(counts, x_var=unseen_frac) #random split
        # save split
        make_dir_if_needed(index_dir)
        for i, df in enumerate(dataframes):
            df = pd.DataFrame(index=df.index).reset_index()
            df.to_csv(join(index_dir,f'index_{i}.csv'))
        pd.DataFrame(dataframes[0].variant.cat.categories).to_csv(join(index_dir, 'categories.csv')) #save category order
        in_shape = dataframes[0].shape[1]-3
        model = config.model_class(
                MLP(input_shape=in_shape, inner_shape=args.shape, dropout=args.dropout,
                    output_shape=args.embed_dim,),
            normalize= not args.no_norm_embeds,
            #task-specific kwargs
            n_class = dataframes[0]['variant'].nunique() # should be equal to nb of codes 
                            ).to(ctx.device)
        run_meta = {
            'i':0,
        }
    train, test_seen, test_unseen = make_loaders(
        *dataframes, batch_size=args.batch_size, n_workers=args.n_workers, 
        pos_frac = args.positive_fraction, dataset_class=config.dataset_class,
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
    train_model(train, test_seen, test_unseen, model, run_meta, model_file, meta_file, 
                loss_fn, optimizer=optimizer, scheduler=scheduler, 
                n_epoch=args.n_epochs,
                )
    
def test(args):
    print('---RUNNING TEST--- ')
    args.data_path = None
    args.restart = False
    make_dir_if_needed(run_dir)
    counts = pd.read_csv('/home/lguirardel/data/perturb_comp/data/KRAS_test.csv', index_col=0)

    main(args, counts, unseen_frac=0)
    

if __name__ == '__main__':
    parser = ArgumentParser('''Train and evaluate a contrastive model on Ursu et al. data''')
    parser.add_argument('data_path', help='Path to data directory. ')
    parser.add_argument('run_name',)

    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--load-split',metavar='RUN', help='If passed, load split fron given run. Use to compare models on the same data')

    parser.add_argument('--loss', choices=[*loss_dict.keys()], default='standard',
                        help='''standard loss : $y ||e_1 - e_2||^2_2 + (1-y) max(||e_1 - e_2||_2 -m, 0)^2 $''')
    parser.add_argument('-m','--margin',default=1, type=float, help='Contrastive loss margin parameter')
    parser.add_argument('-a','--alpha',default =0, type=float, help='L2 embedding regularization')
    parser.add_argument('-d','--dropout',default=0.2, type=float, help='Dropout frequency')
    parser.add_argument('-w','--weight-decay',default=1e-2, type=float, help='Weight decay parameter')
    parser.add_argument('--batch-size',default=128, help='Batch size', type=int)
    parser.add_argument('--positive-fraction',default=0.5, help='Fraction of positive training samples', type=float)
    parser.add_argument('-n','--n-epochs', metavar='N', default=600, type=int, help='Number of epochs to run')
    parser.add_argument('--split-synon',action='store_true', 
                        help='If not passed, group all WT-like variants in the same class')
    parser.add_argument('--no-norm-embeds',action='store_true',
                        help='If passed, do not rescale emebeddings to unit norm')
    # scheduler lr args
    sched_args = parser.add_argument_group('Learning rate scheduler arguments')
    sched_args.add_argument('--lr',type=float, default=1e-3, )
    sched_args.add_argument('--scheduler', choices=['plateau','restarts'], default='plateau',
                            help="plateau : reduce lr on plateau\nrestarts : cosine annealaing with warm restarts")
    sched_args.add_argument('--patience',type=int,help='Patience for reduce lr on plateau', default=40)
    sched_args.add_argument('--cosine-t',type=int,help='Period for cosine annealing', default=100)

    parser.add_argument('-c', '--config-file', is_config_file_arg=True, help='add config file')
    parser.add_argument('--shape', type=int, nargs='+', help = 'MLP shape', default=[100, 100])
    parser.add_argument('--embed-dim', type=int, default=20 ,help='Embedding dimension') # Ursu et al first project in 50 dim, but only use the 20 first ones for sc-eVIP
    parser.add_argument('--n-workers', default=0, type=int, help='Number of workers for datalaoding')
    parser.add_argument('--knn', default=3, type=int, help='Number of neighbors for knn scoring')
    parser.add_argument('--overwrite', action='store_true', help='Do not prompt for confirmation if model already exists')
    parser.add_argument('--task',choices=[*config_dict.keys()], help='Type of learning task to optimize', default='siamese')
    parser.add_argument('--run-test', help='Run test on reduced data', action='store_true')
    parser.add_argument('--cpu', help='Run on cpu even if gpu is available', action='store_true')
    
    args = parser.parse_args()

 
    run_dir = join('models',args.run_name) if not args.run_test else join('models', '_test')
    run_name = args.run_name if not args.run_test else 'TEST' 


    # run test
    if args.run_test:
        test(args)
        print('Test concluded.')
        sys.exit()

    # argument compatibility check
    if args.restart:
        sources = parser._source_to_settings
        for arg in ['shape','embed_dim']:
            if sources['command_line'].get(arg) is not None:
                warn(f'Warning : restart : {arg} will be ignored', )
        if args.load_split is not None:
            warn('Argument load_split will be ignored in favor of split saved for this model previous instance')
    if (not args.no_norm_embeds and args.alpha ) or args.task == 'cycle-classifier':
        warn('Embedding norm penalty parameter alpha is nonzero while embeddings are normalized.')
    if args.task in ['classifier'] and args.alpha : # for cycle classifier, alpha is the cycle/variant weight
        raise NotImplementedError('Nonzero embedding norm penalty parameter alpha is not compatible with a classification task.')
    
    # safety overwriting check
    if os.path.exists(join(run_dir, 'config.ini')) and not args.restart and not args.overwrite:
        check = input(f'{run_dir} already exists, are you sure you want to overwrite ? [Y/n] ')
        if check.lower() in ['n','no']:
            print('Exiting.')
            sys.exit()
        else:
            print('Proceeding.')
    make_dir_if_needed(run_dir)
    # save args to file
    parser.write_config_file(args, [join(run_dir, 'config.ini')], exit_after=False)
    # print(parser._source_to_settings)
    print()
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu' 
    print(f'Using {device}.')
    paths = get_paths(args.data_path)
    print(f'Loading data from {args.data_path}...', flush=True)
    counts = load_data(*paths, group_wt_like= not args.split_synon,)

    ctx = Context(device, run_dir, run_name, task=args.task, k_nn=args.knn)
    main(args, counts)
    
