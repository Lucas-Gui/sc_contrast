from data_utils import *
from models import *
from contrastive_data import make_loaders

from argparse import ArgumentParser
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



loss_dict : Dict[str, Type[ContrastiveLoss]] = {
    'standard':SiameseLoss
}

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

        
def train_model(train, test_seen, test_unseen, model,
                 loss_fn, device, margin, n_epoch=100, run_name='run'):
    optimizer = torch.optim.AdamW(model.parameters())
    bar = tqdm(range(n_epoch), position=0)
    writer = SummaryWriter(join('runs',run_name))
    for i in bar:
        loss_train = train_loop(train, model, loss_fn, optimizer, device)
        writer.add_scalar('train/loss',np.mean(loss_train), i)
        loss_seen, acc_seen = test_loop(test_seen, model, loss_fn, device, margin=margin)
        loss_unseen, acc_unseen = test_loop(test_unseen, model, loss_fn, device, margin=margin)
        writer.add_scalar('test/loss_unseen',np.mean(loss_unseen), i)
        writer.add_scalar('test/loss_seen',np.mean(loss_seen), i)
        writer.add_scalar('test/acc_unseen',np.mean(acc_unseen), i)
        writer.add_scalar('test/acc_seen',np.mean(acc_seen), i)
        torch.save(model, join('models', run_name +  '.model.pkl'))
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

def test_loop(test:DataLoader, model:nn.Module, loss_fn, device, margin) -> Tuple[Tensor]:
    model.eval()
    loss_l = []
    acc_l = []
    with torch.no_grad():
        for x,y in tqdm(test, position=1, desc='Testing loop', leave=False):
            x = (x_i.to(device) for x_i in x)
            y = y.to(device)
            embeddings = model.forward(*x)
            loss:Tensor = loss_fn(*embeddings, y)
            loss_l.append(loss)
            acc_l.append(accuracy(*embeddings, y, margin))
    loss_l = [t.detach().cpu().item() for t in loss_l]
    acc_l = [t.detach().cpu().item() for t in acc_l]
    return loss_l, acc_l

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}.')
    paths = get_paths(args.data_path)
    print(f'Loading data from {args.data_path}...', flush=True)
    counts = load_data(*paths)
    dataframes = split(counts)
    train, test_seen, test_unseen = make_loaders(*dataframes, batch_size=args.batch_size,  )
    in_shape = next(iter(train))[0][0].shape[1]
    model = Siamese(MLP(input_shape=in_shape)).to(device)
    print(model)
    loss_fn = loss_dict[args.loss](margin=args.margin, alpha=args.alpha)
    train_model(train, test_seen, test_unseen, model, loss_fn, device, 
                margin=args.margin, run_name=args.run_name)


if __name__ == '__main__':
    parser = ArgumentParser('''Train and evaluate a contrastive model on Ursu et al. data''')
    parser.add_argument('data_path', help='Path to data directory. Should contain')
    parser.add_argument('--loss', choices=[*loss_dict.keys()], default='standard',
                        help='''standard loss : $y ||e_1 - e_2||^2_2 + (1-y) max(||e_1 - e_2||_2 -m, 0)^2 $''')
    parser.add_argument('-m','--margin',default=1, type=float, help='Contrastive loss margin parameter')
    parser.add_argument('-a','--alpha',default=1e-2, type=float, help='L2 embedding regularization')
    parser.add_argument('--batch-size',default=128, help='Batch size', type=int)
    parser.add_argument('run_name',)
    args = parser.parse_args()
    main(args)
    