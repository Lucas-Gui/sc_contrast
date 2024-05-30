'''Run models in parallel with shared data'''
from argparse import ArgumentParser, Namespace
import torch
import torch.multiprocessing as mp # allow tensor sharing
from torch import Tensor
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from os.path import join
from time import sleep
from typing import *
from configparser import ConfigParser

from hyperparam import PARAM_LIST, CONST_PARAMS, sample
from main import Context, make_data, main, get_counts, make_dir_if_needed
from contrastive_data import Data

## CHANGE hyperparam.py TO CHANGE EXPERIMENT PARAMETERS

## MULTIPROCESSING CODE
Task = Tuple[Namespace, Data, Context]

def worker(name, worker_id, queue : mp.Queue):
    '''Worker process that runs a model and sends back the results'''
    log = open(f'logs/{name}/{worker_id}.log', 'w')
    task : Task
    for task in iter(queue.get, None): # iterates until None is sent
        args, data, ctx = task 
        t = datetime.now()
        print(f'Starting task {ctx.run_name} on worker {worker_id} at time {t.strftime("%d/%m %H:%M")}.')
        log.write(f'Task {ctx.run_name}\n')
        try : 
            with redirect_stderr(log), redirect_stdout(log):
                main(args, data, ctx, )
        except Exception as e:
            print(e)
            raise RuntimeError(f"""Error in task {ctx.run_name} on worker {worker_id}. 
                    See logs/{ctx.run_name}/{worker_id}.log for details.""")
    log.close()

def make_task(param_list, const_params, data, i:int, main_ctx:Context,)->Task:
    '''Create the args namespace and the context and return a task for workers.
    Create the necessary directory and files.'''
    param_dict = sample(param_list, const_params=const_params) # sample hyperparameters
    args = Namespace(**param_dict, name = f"{main_ctx.run_name}_{i}")
    run_dir = join(main_ctx.run_dir, f"{main_ctx.run_name}_{i}")
    # creating model dire and saving config
    make_dir_if_needed(run_dir)
    config = ConfigParser().read_dict({'':param_dict})
    with open(join(run_dir, 'config.ini'), 'w') as f:
        config.write(f)
    ctx = Context(
        main_ctx.device, 
        run_dir=run_dir,
        run_name= f"{main_ctx.run_name}_{i}",
        verbosity = 1,
        task=args.task, k_nn=args.knn,
        index_dir=main_ctx.index_dir,
        model_file=join(run_dir, 'model.pkl'),
        meta_file=join(run_dir, 'meta.json')
    )
    return (args, data, ctx)

def main_f(args, data, main_ctx:Context):
    '''Main function that creates workers and sends them tasks'''
    queue = mp.Queue(args.n_workers+1)
    # initialize queue
    for i in range(args.n_workers):
        task = make_task(PARAM_LIST, CONST_PARAMS, data, i, main_ctx)
        queue.put(task)
    # create workers
    workers = [
        mp.Process(target=worker, args=(main_ctx.run_name,i,queue,)) 
            for i in range(args.n_workers)
            ]
    # start workers
    for w in workers:
        w.start()
    # keep generating tasks until all are done
    for i in range(args.n_workers, args.n_models):
        sleep(0.1)
        if not queue.empty():
            task = make_task(PARAM_LIST, CONST_PARAMS, data, i, main_ctx)
            queue.put(task) # copies everything except tensors, hopefully
    # send stop signal
    for i in range(args.n_workers):
        queue.put(None)
    # wait for workers to finish
    for w in workers:
        w.join()

## MAIN
if __name__ == '__main__':
    mp.set_start_method('spawn') # this or forkserver required for CUDA

    parser = ArgumentParser()
    parser.add_argument('data_path', help='Path to data directory. ')
    parser.add_argument('name')
    parser.add_argument('--filter-variants', metavar='FILE', help='Path to file with variants to include. If not passed, all variants are included', default = None)
    parser.add_argument('--data-subset', default='processed', choices=['processed','raw','filtered'], help='Data version to use')
    parser.add_argument('--group-synon',action='store_true', 
                        help='If passed, group all synonymous variants in the same class')
   # experiment control args
    parser.add_argument('-n','--n-workers', help='Number of model to train in parallel',type=int, default=8)
    parser.add_argument('-N', '--n-models', help='Total number of models to train', default=1000, type=int)
    parser.add_argument('--i0', help='Index of first model to train (useful to continue previous experiments)', default=0, type=int)
    parser.add_argument('--overwrite', help='Overwrite previous experiments', action='store_true')
    
    args = parser.parse_args()

    parent_model_dir = join('models', args.name)
    make_dir_if_needed(parent_model_dir)
    make_dir_if_needed(join('runs', args.name))
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    print(f'Using {device}.')
    counts = get_counts(args)

    main_ctx = Context(
        device, parent_model_dir, run_name=args.name,
        verbosity = 1,
        index_dir = join(parent_model_dir, 'split'),
    )
    data = make_data(args, counts, main_ctx)