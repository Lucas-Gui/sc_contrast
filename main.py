from data_utils import *
from models import *
from contrastive_data import make_loaders

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



loss_dict : Dict[str, Type[ContrastiveLoss]] = {
    'standard':SiameseLoss
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

def load_split(index_dir, counts):
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
            writer.add_scalars(f'{main_tag}_{tag}', metric_or_dict, i)
        else:
            writer.add_scalar(f'{main_tag}_{tag}', metric_or_dict, i)


        
def train_model(train, test_seen, test_unseen, model, model_args:Dict,
                 loss_fn, device, margin, n_epoch=10_000, run_name='run',
                 lr=1e-3
                 ):
    i_0 = model_args['restart']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    bar = tqdm(range(i_0, i_0+n_epoch), position=0)
    writer = SummaryWriter(join('runs',run_name))
    for i in bar:
        bar.set_postfix({'i':i})
        loss_train = train_loop(train, model, loss_fn, optimizer, device)
        writer.add_scalar('train/loss',np.mean(loss_train), i)
        metrics_seen =  test_loop(test_seen, model, loss_fn, device, margin=margin)
        write_metrics(metrics_seen, writer, 'test/seen',i)
        if test_unseen is not None:
            metrics_unseen = test_loop(test_unseen, model, loss_fn, device, margin=margin)
            write_metrics(metrics_unseen, writer, 'test/unseen',i)
        torch.save(model, join('models', run_name +  '.model.pkl'))
        model_args['restart'] = i
        with open(join('models',run_name+'.meta.json'), 'w') as file:
            json.dump(model_args, file, sort_keys=True, indent=2)
    writer.flush()



def train_loop(train:DataLoader, model:nn.Module, loss_fn:ContrastiveLoss, optimizer:torch.optim.Optimizer, device)-> Tensor:
    model.train()
    loss_l = []
    for x,y in tqdm(train, position=1, desc='Training loop', leave=False):
        x = (x_i.to(device) for x_i in x)
        y = y.to(device)
        embeddings = model.forward(*x)
        loss:Tensor = loss_fn.forward(*embeddings, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_l.append(loss)
    return [t.detach().cpu().item() for t in loss_l]

def test_loop(test:DataLoader, model:nn.Module, loss_fn:ContrastiveLoss, device, margin, ) -> Dict[str, float|Dict]:
    model.eval()
    loss_l = []
    acc_l = []
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
            acc_l.append(accuracy(*embeddings, y, margin))
            y_l.append(y)
            e1, e2 = embeddings
            d_l.append(torch.norm(e1-e2, p=2, dim=-1)) #TODO : change to accomodate triplet loss
            norm_l.append((torch.norm(e1, dim=-1) + torch.norm(e2, dim=-1))*loss_fn.alpha)
    d = torch.concat(d_l)
    y = torch.concat(y_l)
    norm = torch.concat(norm_l)
    metrics = {
        'dist':{
            'pos' : ((d*y).sum()/y.sum()).item(),
            'neg' : ((d* ~y).sum()/(~y).sum()).item()
            },
        'l2_penalty':norm.mean().item(),
        'loss' : np.mean( [t.detach().cpu().item() for t in loss_l]),
        'acc' : np.mean([t.detach().cpu().item() for t in acc_l]),
    }
    return metrics

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}.')
    paths = get_paths(args.data_path)
    print(f'Loading data from {args.data_path}...', flush=True)
    counts = load_data(*paths)
    index_dir = join('models', args.run_name+'split')
    model_file = join('models',args.run_name+'.model.pkl')
    if args.restart : #load model and split
        dataframes = load_split(index_dir, counts)
        print(f'Loading model from {model_file}')
        model = torch.load(model_file)
        with open(join('models', args.run_name+'.meta.json'), 'r') as reader:
            model_args = json.load(reader)

    else:
        dataframes = split(counts) #random split
        # save split
        make_dir_if_needed(index_dir)
        for i, df in enumerate(dataframes):
            df = pd.DataFrame(index=df.index).reset_index()
            df.to_csv(join(index_dir,f'index_{i}.csv'))
        in_shape = dataframes[0].shape[1]-2
        model = Siamese(MLP(input_shape=in_shape)).to(device)
        model_args = {
            'restart':0,
        }
    
    train, test_seen, test_unseen = make_loaders(*dataframes, batch_size=args.batch_size,  )
    in_shape = next(iter(train))[0][0].shape[1]

    print(model)
    loss_fn = loss_dict[args.loss](margin=args.margin, alpha=args.alpha)
    train_model(train, test_seen, test_unseen, model, model_args, loss_fn, device, 
                margin=args.margin, run_name=args.run_name, lr=args.lr)


if __name__ == '__main__':
    parser = ArgumentParser('''Train and evaluate a contrastive model on Ursu et al. data''')
    parser.add_argument('data_path', help='Path to data directory. Should contain')
    parser.add_argument('--loss', choices=[*loss_dict.keys()], default='standard',
                        help='''standard loss : $y ||e_1 - e_2||^2_2 + (1-y) max(||e_1 - e_2||_2 -m, 0)^2 $''')
    parser.add_argument('-m','--margin',default=1, type=float, help='Contrastive loss margin parameter')
    parser.add_argument('-a','--alpha',default=1e-2, type=float, help='L2 embedding regularization')
    parser.add_argument('--batch-size',default=128, help='Batch size', type=int)
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--lr',type=float, default=1e-3, )
    parser.add_argument('run_name',)
    args = parser.parse_args()
    main(args)
    