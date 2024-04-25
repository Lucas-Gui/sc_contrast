import numpy as np
from typing import *
from argparse import ArgumentParser
import main
import asyncio
from pprint import pprint
from datetime import datetime

rng = np.random.default_rng()


def logodds(x):
    return np.log10(x/(1-x))

 ## Version 1 : implement myself
class _NumParam():
    def __init__(self, name, min_val, max_val, mode : Literal['lin','log', 'logodds', 'pow2'] = 'lin', 
                 type:Type=None ) -> None:
        self.name = name
        self.min = min_val
        self.max = max_val
        self.mode = mode
        self.type = type
        if mode == 'pow2':
            self.type=int


# class NumParamRange(_NumParam):
#     @property
#     def values(self, n):
#         if self.mode == 'lin':
#             return np.linspace(self.min, self.max, n)
#         elif self.mode == 'log':
#             x = np.linspace(np.log10(self.min), np.log10(self.max), n)
#             return 10**x
#         else:
#             raise ValueError(f'mode needs to be lin or log, got "{self.mode}"')
        
class NumParamSampler(_NumParam):
    def sample(self):
        if self.mode == 'lin':
            x= rng.uniform(self.min, self.max)
        elif self.mode == 'log':
            x = rng.uniform(np.log10(self.min), np.log10(self.max))
            x= 10**x
        elif self.mode == 'logodds': # for variable which are ratios
            u = rng.uniform(logodds(self.min), logodds(self.max))
            x= 10**u / (1 + 10**u)
        elif self.mode == 'pow2':
            x = rng.uniform(np.log2(self.min), np.log2(self.max))
            x = 2**int(x)

        else :
            raise ValueError(f'mode needs to be lin, log, or symlog got "{self.mode}"')
        return self.type(x) if self.type is not None else x
        
class CatParamSampler():
    def __init__(self, name, values) -> None:
        self.name = name
        self.values = values

    def sample(self):
        return rng.choice(self.values)
    
class ConstParamSampler():
    '''Placeholder for convenience'''
    def __init__(self, name, value) -> None:
        self.name = name
        self.value = value

    def sample(self):
        return self.value
    
class ShapeSampler():
    def __init__(self, name, n_layer_max, n_neurons_min, n_neurons_max, ) -> None:
        self.name = name
        self.n_layer_max = n_layer_max
        self.n_neurons_min = n_neurons_min
        self.n_neurons_max = n_neurons_max
    def sample(self):
        n = rng.integers(self.n_neurons_min, self.n_neurons_max+1, endpoint=True, dtype=int)
        l = rng.integers(1, self.n_layer_max+1, dtype=int )
        return ' '.join([str(n)] * l)

# PARAM_LIST = [
#     NumParamSampler('lr', 1e-5, 1e-1, 'log'),
#     CatParamSampler('scheduler', ['restarts','plateau']),
#     NumParamSampler('patience',10,200, 'log', type=int),
#     NumParamSampler('cosine-t',200, 600, 'lin', type=int),
#     CatParamSampler('loss',[*main.loss_dict.keys()]),# will have to reset to standard if we are not choosing siamese
#     NumParamSampler('margin',1e-3, 1-1e-3, 'logodds'),
#     # NumParamSampler('alpha', min_val=) # alpha = 0 since we normalize
#     NumParamSampler('dropout', min_val=1e-1, max_val=0.5,mode='log'),
#     NumParamSampler('weight-decay', min_val=1e-4, max_val=1,mode='log'),
#     NumParamSampler('batch-size', min_val=16, max_val=1024, mode='pow2'),
#     NumParamSampler('positive-fraction', 0.01, 0.99, 'logodds', ),
#     ShapeSampler('shape',5, 20, 200),
#     ConstParamSampler('embed-dim', 20),
#     CatParamSampler('task',[*main.config_dict.keys()])

# ]

# PARAM_LIST = [ # ,  siamese only
#     NumParamSampler('lr', 1e-5, 1e-1, 'log'),
#     CatParamSampler('scheduler', ['restarts','plateau']),
#     NumParamSampler('patience',10,500, 'log', type=int),
#     NumParamSampler('cosine-t',50, 600, 'lin', type=int),
#     ConstParamSampler('loss','standard'),# will have to reset to standard if we are not choosing siamese
#     # NumParamSampler('margin',0.5, 4, 'log'),
#     # NumParamSampler('alpha', min_val=) # alpha = 0 since we normalize
#     NumParamSampler('dropout', min_val=1e-3, max_val=0.5,mode='log'),
#     NumParamSampler('weight-decay', min_val=1e-5, max_val=10,mode='log'),
#     NumParamSampler('batch-size', min_val=128, max_val=2048, mode='pow2'),
#     NumParamSampler('positive-fraction', 0.4, 0.6, 'logodds', ),
#     ShapeSampler('shape',4, 20, 1_000),
#     ConstParamSampler('embed-dim', 20),
#     ConstParamSampler('task', 'siamese'),
#     # CatParamSampler('mil-mode',['mean', 'attention']),
#     # NumParamSampler('bag-size', 1, 100, 'log', type=int),
# ]

PARAM_LIST = [ # ,  siamese only
    NumParamSampler('lr', 1e-4, 1e-1, 'log'),
    ConstParamSampler('scheduler', 'restarts'),
    # NumParamSampler('patience',10,500, 'log', type=int),
    NumParamSampler('cosine-t',50, 600, 'lin', type=int),
    ConstParamSampler('loss','standard'),# will have to reset to standard if we are not choosing siamese
    # NumParamSampler('margin',0.5, 4, 'log'),
    # NumParamSampler('alpha', min_val=) # alpha = 0 since we normalize
    NumParamSampler('dropout', min_val=1e-3, max_val=0.1,mode='log'),
    NumParamSampler('weight-decay', min_val=1e-5, max_val=1,mode='log'),
    ConstParamSampler('batch-size', 512),
    # NumParamSampler('positive-fraction',0.4,0.6, 'logodds', ),
    ShapeSampler('shape',4, 100, 1_000),
    ConstParamSampler('embed-dim', 20),
    CatParamSampler('task',[*main.config_dict.keys()]),
    ConstParamSampler('n-epochs', 200),
    # FOR NABID DATA
    ConstParamSampler('unseen-fraction', 0),
    ConstParamSampler('data-subset', 'filtered'),
    # CatParamSampler('mil-mode',['mean', 'attention']),
    # NumParamSampler('bag-size', 1, 100, 'log', type=int),
]
def make_task(args)-> str:
    pass


def sample(params, ):
    d = {}
    for p in params :
        d[p.name] = p.sample()
    # additional processing
    if d['task'] != 'siamese':
        d['loss'] = 'standard'
    return d

async def worker(name, worker_name, gen, path, load_split = True, overwrite = False):
    log = open(f'logs/{name}/{worker_name}.log', 'w')
    for task_name, params, i in gen:
        # creating the command
    #python main.py ~/data/nabid_data/p53pilot/Data/RawData/filtered_feature_bc_matrix/wrangled/ test_nabid --dest-name nabid 

        cmd =  (f'python main.py {path} {task_name} --dest-name {name} --verbose 1 ' 
                # +'--no-norm-embeds ' # !! to change if wanted
                +' '.join([f'--{arg} {param}' for arg, param in params.items() ]))
        if overwrite:
            cmd += ' --overwrite'
        if i > 0 and load_split:
            cmd += f' --load-split {name}/{name}_0' # this relies on the fact that the name is name_i
            if i <= int(worker_name):
                await asyncio.sleep(10) # wait some time to allow first worker to split the data on first task
        # logging
        t = datetime.now()
        print(f'Starting task {task_name} on worker {worker_name} at time {t.strftime("%d/%m %H:%M")}.')
        log.write(f'Task {task_name}\n')
        log.write(cmd+'\n')

        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=log,
            stderr=log
        )
        exit_code = await process.wait()
        log.write('\n')
        log.flush()
        dt = (datetime.now() - t)
        print(f'Finished task {task_name} on worker {worker_name} after {dt.seconds // 60 }m {dt.seconds % 60}s.')
        if exit_code != 0:
           raise RuntimeError(f'Error in task {task_name} on worker {worker_name} : exit code {exit_code}.')
    log.close()

async def main_f(args):
    #TODO : create directories for everything so that first workers don't try to do it at the same time
    #TODO : add a checkpoint to wait for first worker to split data before starting the others
    gen = ((args.name+f"_{i}", sample(PARAM_LIST), i) for i in range(args.i0, args.i0+args.n_models))
    workers = [asyncio.create_task(worker(args.name, str(i), gen, args.path, overwrite=args.overwrite)) for i in range(args.n_workers)]
    await asyncio.gather(*workers)

if __name__ == '__main__':
    parser = ArgumentParser('''Train models in parallel to explore hyperparameter space''')
    parser.add_argument('path', help='Path to data directory. ')
    parser.add_argument('name')
    parser.add_argument('-n','--n-workers', help='Number of model to train in parallel',type=int, default=8)
    parser.add_argument('-N', '--n-models', help='Total number of models to train', default=1000, type=int)
    parser.add_argument('--i0', help='Index of first model to train (useful to continue previous experiments)', default=0, type=int)
    parser.add_argument('--overwrite', help='Overwrite previous experiments', action='store_true')
    
    args = parser.parse_args()

    main.make_dir_if_needed(f'logs/{args.name}')
    main.make_dir_if_needed(f'models/{args.name}')
    main.make_dir_if_needed(f'runs/{args.name}')

    

    asyncio.run(main_f(args))


    