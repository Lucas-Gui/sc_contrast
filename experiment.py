'''Run models in parallel with shared data'''
from argparse import ArgumentParser, Namespace
from pprint import pprint
import torch
import torch.multiprocessing as mp # allow tensor sharing
from torch.cuda import OutOfMemoryError
from torch import Tensor
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from os.path import join
from os import getpid
from time import sleep
from typing import *
from configparser import ConfigParser

from hyperparam import PARAM_LIST, CONST_PARAMS, sample
from main import Context, DataSource, main, get_counts, make_dir_if_needed, make_parser, split_data
from contrastive_data import Data, SlicedData
from enum import Enum
from psutil import Process

# args that are defined by user during command invocation 
# and should be passed to workers
OVERRIDE_ARGS = [ 'overwrite', 'restart'] # 'load_split','unseen_frac',
class SplitPolicy(Enum):
    SHARED_RANDOM = 1 # save a random split to experiment directory and load it for all models
    LOAD_ALL = 2 # load a split from a given directory for all models
    LOAD_EACH = 3 # load a split from the model directory
    RANDOM = 4 # create a new random split for each model
    K_FOLD = 5 # use k-fold cross-validation


## CHANGE hyperparam.py TO CHANGE EXPERIMENT PARAMETERS

## MULTIPROCESSING CODE
Task = Tuple[Namespace, Data|List[Data], Context,]

def worker_f(name, worker_id, queue : "mp.Queue[Task]"):
    '''worker function that runs a model and sends back the results'''
    log = open(f'logs/{name}/{worker_id}.log', 'w')
    for args, data, ctx in iter(queue.get, None): # iterates until None is sent
        t = datetime.now()
        print(f'Starting task {ctx.run_name} on worker {worker_id} (pid {getpid()}) at {t.strftime("%d/%m %H:%M")}.',
            f'Logging to logs/{name}/{worker_id}.log',)
        log.write(f'Task {ctx.run_name}\n')
        log.write(t.strftime("%d/%m %H:%M")+'\n')
        log.write(str(args)+'\n')
        try :
            with redirect_stderr(log), redirect_stdout(log):
                main(args, data, ctx,  ) 
        except OutOfMemoryError as e:
            log.write(f"Out of memory error in task {ctx.run_name}.")
            print(f"Out of memory error in task {ctx.run_name} : terminating worker {worker_id}.")
            break
        # except Exception as e:
        #     print(e)
        #     raise RuntimeError(f"""Error in task {ctx.run_name} on worker {worker_id}. 
        #             See logs/{name}/{worker_id}.log for details.""")
        log.write("\n")
    log.close()

def make_task(param_list, const_params, data, i, main_ctx:Context, split_policy, )->Task:
    '''
    Create the args namespace and the context and return a task for workers.
    Create the necessary directory and files.
    Handles the splitting policy by modifying the load_split argument.
    '''
    # print(f"{split_policy=}")
    run_dir = join(main_ctx.run_dir, f"{main_ctx.run_name}_{i}")
    # creating model dire and saving config
    make_dir_if_needed(run_dir)
    # sampling run arguments and populating args Namespace
    param_dict = sample(param_list, const_params ) # sample hyperparameters
    match split_policy:
        case SplitPolicy.SHARED_RANDOM | SplitPolicy.LOAD_ALL | SplitPolicy.K_FOLD:
            param_dict['load_split'] = main_ctx.index_dir
        case SplitPolicy.LOAD_EACH:
            param_dict['load_split'] = f"models/{main_ctx.run_name}/{main_ctx.run_name}_{i}/split"
        case SplitPolicy.RANDOM:
            pass # already None
    args = make_parser().parse_args('N/A N/A') 
    for key, value in param_dict.items(): # update args with hyperparameters
        setattr(args, key, value)
    with open(join(run_dir, 'config.ini'), 'w') as f:
        param_cfg = ConfigParser()
        # pprint(param_dict)
        param_cfg.read_dict({'args':(param_dict)})
        param_cfg.write(f)
    ctx = Context(
        main_ctx.device, 
        run_dir=run_dir,
        run_name= f"{main_ctx.run_name}_{i}",
        verbosity = main_ctx.verbosity,
        task=args.task, k_nn=args.knn,
        index_dir=join(run_dir, 'split'),
        model_file=join(run_dir, 'model.pkl'),
        meta_file=join(run_dir, 'meta.json')
    )
    return (args, data, ctx, )

def make_n_tasks(param_list, const_params, data:List[List[SlicedData]|Data], i, main_ctx, split_policy)-> List[Task]:
    '''
    Make n tasks, according to split policy.
    If split policy is not K_FOLD, make only one task. Otherwise, make k tasks.
    '''
    if split_policy == SplitPolicy.K_FOLD:
        return [make_task(param_list, const_params, data[k], f"{i}_f{k+1}", main_ctx, split_policy) for k in range(args.k_fold)]
    else:
        return [make_task(param_list, const_params, data[0], i, main_ctx, split_policy)]

def main_f(args, data, main_ctx:Context, split_policy:SplitPolicy):
    '''Create workers and distribute tasks'''
    queue = mp.Queue(args.n_workers+1)
    i0 = args.i0
    for arg in OVERRIDE_ARGS:
        try :
            _ = CONST_PARAMS[arg]
        except KeyError:
            pass
        else:
            print("Warning : overriding ", arg, getattr(args, arg), "from CONST_PARAMS in hyperparam.py.")
        CONST_PARAMS[arg] = getattr(args, arg)

    # create workers
    workers = [
        mp.Process(target=worker_f, args=(main_ctx.run_name,i,queue,), name=f'worker_{i}') 
            for i in range(args.n_workers)
            ]
    # start workers
    for w in workers:
        w.start() # workers will wait for tasks in queue
    # keep generating tasks until all are done
    for i in range(i0, i0+args.n_models):
        tasks = make_n_tasks(PARAM_LIST, CONST_PARAMS, data, i, main_ctx, split_policy=split_policy)
        for task in tasks:
            # put task in queue, blocking until queue has space
            queue.put(task, block=True, timeout=None) # copies everything except tensors
        # check if processes are alive
        for w in workers:
            if not w.is_alive():
                print(f"Worker {w.name} is dead. Joining.")
                w.join()
    # send stop signal
    for _ in range(args.n_workers):
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
    parser.add_argument('--cooper', action='store_true', help='If passed, use Cooper data')
    # data args --> get_counts
    parser.add_argument('--data-subset', default='processed', choices=['processed','raw','filtered'], help='Data version to use')
    parser.add_argument('--group-synon',action='store_true', 
                        help='If passed, group all synonymous variants in the same class')
    # preprocessing args --> get_counts --> _preprocess_counts
    parser.add_argument('--filter-cells', action='store_true', help='If passed, filter cells based on counts, number of expressed genes, and mitochondrial counts.')
    parser.add_argument('--filter-variants', metavar='FILE', help='Path to file with variants to include. If not passed, all variants are included', default = None)
    parser.add_argument('--log1p', action='store_true', help='If passed, log1p transform the data')
    parser.add_argument('--cell-min', metavar='N', type=int, help='Minimum number of cells under which variants are discarded', default=10)
    # split args --> main --> split_data
    # let's define them in the experiments parameters for now and see if we need to change that
    # parser.add_argument('--load-split',metavar='RUN', help='If passed, load split fron given run. Use to compare models on the same data')
    parser.add_argument('--restart', action='store_true', help="If passed, reload existing models. Will raise an error if models don't exist")

    group = parser.add_argument_group('Split policy. If no argument is passed, uses k-fold cross-validation.')
    group = group.add_mutually_exclusive_group()
    group.add_argument('--k-fold',metavar='k', default=5, type=int, help='Default. Use K_FOLD split policy with k splits. For each set of hyperparameters, train k models on k splits.')
    group.add_argument('--load-all-split-from', metavar='RUN', default=None, nargs='?', const='DEFAULT',
        help='If passed, load all splits from given directory. Use to compare models on the same data. If no directory is passed, uses default run directory.',
                       )
    group.add_argument('--load-each-split', action='store_true', help='If passed, for each model, load split from its own directory. Use with premade splits.')
    group.add_argument('--random-split', action='store_true', help='If passed, split data randomly separately for all models.')
    group.add_argument('--unseen-frac', type=float, help='Fraction of unseen variants. If passed, use the SHARED_RANDOM split policy and generate a new data split to use for all models.', default=None)
    # experiment control args
    parser.add_argument('-n','--n-workers', help='Number of model to train in parallel',type=int, default=8)
    parser.add_argument('-N', '--n-models', help='Total number of models to train', default=1000, type=int)
    parser.add_argument('--i0', help='Index of first model to train (useful to continue previous experiments)', default=0, type=int)
    parser.add_argument('--overwrite', help='Overwrite previous experiments', action='store_true')
    parser.add_argument('--cpu', action='store_true', help='Use CPU. Default: use GPU if available.')
    
    args = parser.parse_args()

    # if not args.random_split and args.load_all_split_from is None and not args.load_each_split :
    #     setattr(args, 'shared_random_split', True)
    # else:
    #     setattr(args, 'shared_random_split', False)

    parent_model_dir = join('models', args.name)
    make_dir_if_needed(parent_model_dir)
    make_dir_if_needed(join('runs', args.name))
    make_dir_if_needed(f'logs/{args.name}')
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu' 
    print(f'Using {device}.')
    counts = get_counts(args)
    # determine split policy and corresponding index directory
    if args.unseen_frac is not None:
        _index_dir = join('models', args.name, 'split') 
        split_policy = SplitPolicy.SHARED_RANDOM
    elif args.load_all_split_from is not None:
        if args.load_all_split_from == 'DEFAULT':
            _index_dir = join('models', args.name, 'split')
        else:
            _index_dir = args.load_all_split_from
        split_policy = SplitPolicy.LOAD_ALL
    if args.load_each_split:
        _index_dir = ''
        split_policy = SplitPolicy.LOAD_EACH
    elif args.random_split:
        _index_dir = ''
        split_policy = SplitPolicy.RANDOM
    else: # default case
        _index_dir = join('models', args.name, 'split') 
        split_policy = SplitPolicy.K_FOLD
    
    main_ctx = Context(
        device, parent_model_dir, run_name=args.name,
        verbosity = 2,
        index_dir=_index_dir,
        data_source=DataSource.COOPER if args.cooper else None
    )
    data = Data.from_df(counts, device=device)
    # split data if using a shared split
    if split_policy == SplitPolicy.SHARED_RANDOM:
        data = split_data(data, main_ctx, restart=False, load_split_path=None, unseen_frac=args.unseen_frac)
    elif split_policy == SplitPolicy.LOAD_ALL:
        data = split_data(data, main_ctx, restart=False, load_split_path=_index_dir,)
    elif split_policy == SplitPolicy.K_FOLD:
        data = split_data(data, main_ctx, restart=False, load_split_path=None, k_var=args.k_fold)
    else :
        print('Using separated data splits. Watch out for memory usage.')
        data = [data] # make it a list for make_n_tasks
    main_f(args, data, main_ctx, split_policy=split_policy)