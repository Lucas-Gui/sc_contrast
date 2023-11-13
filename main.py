import argparse
from data_utils import *
from models import *
from contrastive_data import make_loaders
from scoring import *

from configargparse import ArgumentParser
import torch 
from torch import Tensor
import torch.nn as nn
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

# config 

loss_dict : Dict[str, Type[ContrastiveLoss]] = {
    'standard':SiameseLoss,
    'lecun':LeCunContrastiveLoss
}


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
              '.variants.csv']:
        l = glob(join(data_dir, '*'+p))
        assert len(l)==1, f"There should be exaclty one match for {join(data_dir, '*'+p)}"
        paths.extend(l)
    return paths

def load_split(index_dir, counts) -> List[pd.DataFrame]:
        i = 0
        dataframes = []
        while os.path.isfile(join(index_dir, f'index_{i}.csv')):
            idx = pd.read_csv(join(index_dir, f'index_{i}.csv'), index_col=1).index
            dataframes.append(counts.loc[idx])
            i+=1
        return dataframes

def write_metrics(metrics:  Dict[str, float|Dict], writer:SummaryWriter, main_tag:str, i):
    for tag, metric_or_dict in metrics.items():
        if isinstance(metric_or_dict, dict):
            writer.add_scalars(f'{main_tag}/{tag}', metric_or_dict, i)
        else:
            if not tag.startswith('_'):
                writer.add_scalar(f'{main_tag}/{tag}', metric_or_dict, i)


        
def train_model(train, test_seen, test_unseen, model, run_meta, model_file, meta_file,
                 loss_fn, margin, device, n_epoch=10_000, 
                 lr=1e-3, weight_decay=0.001, 
                 ):
    i_0 = run_meta['i']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(optimizer)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)
    bar = tqdm(range(i_0, i_0+n_epoch), position=0)
    writer = SummaryWriter(join('runs',run_name))
    best_score = - np.inf
    for i in bar:
        bar.set_postfix({'i':i})
        metrics_train = train_loop(train, model, loss_fn, optimizer, device=device)
        write_metrics(metrics_train, writer, 'train', i)
        metrics_seen =  test_loop(test_seen, model, loss_fn, margin=margin, device=device)
        write_metrics(metrics_seen, writer, 'test_seen',i)
        if metrics_seen['roc'] > best_score:
            best_score = metrics_seen['roc']
            torch.save(model, join(run_dir, 'best_model.pkl'))
            with open(join(run_dir, 'best_score.json'), 'w') as file:
                json.dump({'i':i, 'roc_seen':best_score}, file, sort_keys=True, indent=2)
        scheduler.step(metrics_seen['roc'])

        if test_unseen is not None:
            metrics_unseen = test_loop(test_unseen, model, loss_fn, margin=margin, device=device)
            write_metrics(metrics_unseen, writer, 'test_unseen',i)
        #saving and writing
        torch.save(model, model_file)
        run_meta['i'] = i
        with open(meta_file, 'w') as file:
            json.dump(run_meta, file, sort_keys=True, indent=2)
    writer.flush()



def train_loop(train:DataLoader, model:nn.Module, loss_fn:ContrastiveLoss, optimizer:torch.optim.Optimizer, device)-> Tensor:
    model.train()
    loss_l = []
    y_l = []
    d_l = []
    norm_l = []
    for x,y in tqdm(train, position=1, desc='Training loop', leave=False):
        x = (x_i.to(device) for x_i in x)
        y = y.to(device)
        embeddings = model.forward(*x)
        loss:Tensor = loss_fn.forward(*embeddings, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_l.append(loss)
        y_l.append(y)
        e1, e2 = embeddings
        d_l.append(torch.norm(e1-e2, p=2, dim=-1)) #TODO : change to accomodate triplet loss
        norm_l.append((torch.norm(e1, dim=-1) + torch.norm(e2, dim=-1))*loss_fn.alpha)
    d = torch.concat(d_l).detach().cpu()
    y = torch.concat(y_l).detach().cpu()
    norm = torch.concat(norm_l).detach().cpu()
    metrics = {
        'dist_pos' : ((d*y).sum()/y.sum()).item(),
        'dist_neg' : ((d* ~y).sum()/(~y).sum()).item(),
        'l2_penalty':norm.mean().item(),
        'loss' : np.mean( [t.detach().cpu().item() for t in loss_l]),
        'roc': ROC_score(y, d)[0].item(),
        'lr':optimizer.param_groups[0]['lr']
    }
    return metrics

def test_loop(test:DataLoader, model:nn.Module, loss_fn:ContrastiveLoss, margin, device) -> Dict[str, float|Dict]:
    model.eval()
    loss_l = []
    y_l = []
    d_l = []
    norm_l = []
    with torch.no_grad():
        for x,y in tqdm(test, position=1, desc='Testing loop', leave=False):
            x = (x_i.to(device) for x_i in x)
            y = y.to(device)
            embeddings = model.forward(*x)
            loss:Tensor = loss_fn(*embeddings, y)
            loss_l.append(loss)
            y_l.append(y)
            e1, e2 = embeddings
            d_l.append(torch.norm(e1-e2, p=2, dim=-1)) #TODO : change to accomodate triplet loss
            norm_l.append((torch.norm(e1, dim=-1) + torch.norm(e2, dim=-1))*loss_fn.alpha)
    d = torch.concat(d_l).cpu()
    y = torch.concat(y_l).cpu()
    norm = torch.concat(norm_l).cpu()
    metrics = {
        'dist_pos' : ((d*y).sum()/y.sum()).item(),
        'dist_neg' : ((d* ~y).sum()/(~y).sum()).item(),
        'l2_penalty':norm.mean().item(),
        'loss' : np.mean( [t.detach().cpu().item() for t in loss_l]),
        'roc': ROC_score(y, d)[0].item(),
        '_n_pos' : y.sum().item(),
        '_n_neg' : (~y).sum().item(),
    }
    return metrics



def main(args, counts, unseen_frac = 0.25, device='cuda'):
    print(f"{run_dir=}")

    index_dir = join(run_dir, 'split')
    model_file = join(run_dir,'model.pkl')
    meta_file = join(run_dir, 'meta.json')
    if args.restart : #load model and split
        dataframes = load_split(index_dir, counts)
        print(f'Loading model from {model_file}')
        model = torch.load(model_file)
        with open(meta_file, 'r') as file:
            run_meta = json.load(file) # data that we want to keep between restarts
    else:
        dataframes = split(counts, x_var=unseen_frac) #random split
        # save split
        make_dir_if_needed(index_dir)
        for i, df in enumerate(dataframes):
            df = pd.DataFrame(index=df.index).reset_index()
            df.to_csv(join(index_dir,f'index_{i}.csv'))
        in_shape = dataframes[0].shape[1]-2
        model = Siamese(
            MLP(input_shape=in_shape, inner_shape=args.shape, dropout=args.dropout,
                output_shape=args.embed_dim), normalize=~ args.no_norm_embeds
                            ).to(device)
        run_meta = {
            'i':0,
        }
    print(f"Dataset sizes : "+', '.join(str(df.shape[0]) for df in dataframes))
    train, test_seen, test_unseen = make_loaders(
        *dataframes, batch_size=args.batch_size, n_workers=args.n_workers, 
        pos_frac = args.positive_fraction )
    in_shape = next(iter(train))[0][0].shape[1]

    print(model)
    loss_fn = loss_dict[args.loss](margin=args.margin, alpha=args.alpha)
    train_model(train, test_seen, test_unseen, model, run_meta, model_file, meta_file, 
                loss_fn, device=device,
                margin=args.margin, lr=args.lr, n_epoch=args.n_epochs,
                weight_decay=args.weight_decay
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

    parser.add_argument('--loss', choices=[*loss_dict.keys()], default='standard',
                        help='''standard loss : $y ||e_1 - e_2||^2_2 + (1-y) max(||e_1 - e_2||_2 -m, 0)^2 $''')
    parser.add_argument('-m','--margin',default=1, type=float, help='Contrastive loss margin parameter')
    parser.add_argument('-a','--alpha',default=1e-2, type=float, help='L2 embedding regularization')
    parser.add_argument('-d','--dropout',default=0.2, type=float, help='Dropout frequency')
    parser.add_argument('-w','--weight-decay',default=1e-2, type=float, help='Weight decay parameter')
    parser.add_argument('--batch-size',default=128, help='Batch size', type=int)
    parser.add_argument('--positive-fraction',default=0.5, help='Fraction of positive training samples', type=float)
    parser.add_argument('--lr',type=float, default=1e-3, )
    parser.add_argument('-n','--n-epochs', metavar='N', default=10_000, type=int, help='Number of epochs to run')
    parser.add_argument('--split-wt-like',action='store_true', 
                        help='If not passed, group all WT-like variants in the same class')
    parser.add_argument('--no-norm-embeds',action='store_true',
                        help='If not passed, rescale emebeddings to unit norm')
    
    parser.add_argument('-c', '--config-file', is_config_file_arg=True, help='add config file')
    parser.add_argument('--shape', type=int, nargs='+', help = 'MLP shape', default=[100, 100])
    parser.add_argument('--embed-dim', type=int, default=20 ,help='Embedding dimension')
    parser.add_argument('--n-workers', default=0, type=int, help='Number of workers for datalaoding')
    parser.add_argument('--overwrite', action='store_true', help='Do not prompt for confirmation if model already exists')
    
    parser.add_argument('--run-test', help='Run test on reduced data', action='store_true')
    args = parser.parse_args()

    # GLOBAL VARIABLES
    #This should only contain whatever I would be comfortable setting in a notebook lower namespace
    run_dir = join('models',args.run_name) if not args.run_test else join('models', '_test')# GLOBAL VARIABLE
    run_name = args.run_name if not args.run_test else 'TEST' #GLOBAL VARIABLE #might not be justified


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
    if ~ args.no_norm_embeds and args.alpha :
        warn('Embedding norm penalty parameter alpha is nonzero while embeddings are normalized.')
    
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #do not use as global : this will not work in jupyter
    print(f'Using {device}.')
    paths = get_paths(args.data_path)
    print(f'Loading data from {args.data_path}...', flush=True)
    counts = load_data(*paths, group_wt_like= not args.split_wt_like,)

    main(args, counts)
    