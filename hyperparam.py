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
    def __init__(self, name, min_val, max_val, mode : Literal['lin','log', 'logodds', 'pow2', 'pow2_u'] = 'lin', 
                 type:Type=None ) -> None: #todo : abstract class is not necessary anymore
        self.name = name
        self.min = min_val
        self.max = max_val
        self.mode = mode
        self.type = type
        if mode.startswith('pow2'):
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
    '''
    Sample a number in the range [min, max] according to the mode. Parameters are sampled uniformly, then transformed according to the mode.
    Output variable is always in the (min, max) range.
    mode transforms: 
    - 'lin' : linear sampling (no transform)
    - 'log' : log 
    - 'logodds' : logit transform (for variables that are ratios)
    - 'pow2' : powers of 2
    - 'pow2_u' : uniform in log_2 space, rounded to nearest integer
    '''
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
        elif self.mode == 'pow2_u':
            x = rng.uniform(np.log2(self.min), np.log2(self.max))
            x = int(2**x)
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
    def __init__(self, name, n_layer_min, n_layer_max, n_neurons_min, n_neurons_max, ) -> None:
        '''Sample a shape with n_layer_min to n_layer_max layers and n_neurons_min to n_neurons_max neurons per layer (inclusive).'''
        self.name = name
        self.n_layer_min = n_layer_min
        self.n_layer_max = n_layer_max
        self.n_neurons_min = n_neurons_min
        self.n_neurons_max = n_neurons_max
    def sample(self):
        n = rng.integers(self.n_neurons_min, self.n_neurons_max+1, dtype=int)
        l = rng.integers(self.n_layer_min, self.n_layer_max+1, dtype=int )
        return [n]*l
    
# choose one sampler or another with different probabilities
class ChoiceSampler():
    def __init__(self, name, samplers, probs) -> None:
        self.samplers = samplers
        self.probs = probs
        self.name = name

    def sample(self):
        return rng.choice(self.samplers, p=self.probs).sample()


# _ for experiment.py, - for hyperparam.py
# PARAM_LIST = [
#     ChoiceSampler('projection-shape', [ShapeSampler('', 2, 20, 20), ConstParamSampler('', '')], [2/3, 1/3]),
# ]

PARAM_LIST = [ 
    NumParamSampler('embed_dim', 129, 1024, type=int, mode='pow2_u'),
    CatParamSampler('task', ['classifier', 'batch-supervised']),
    # CatParamSampler('no_norm_embeds', [True, False]), # need to debug why it gives nans
    NumParamSampler('weight_decay', 1e-5, 1e-1, 'log'),
    # CatParamSampler('log1p', [True, False]),
    # ShapeSampler('shape', 1, 10, 512, 512),
]
CONST_PARAMS = { #Parameters that are constant for all models IN EXPERIMENT.PY
    # override the default values
    # arg names should use underscores, not dashes
    # 'task':'classifier',
    # 'no_norm_embeds':True, #comment to normalize embeddings
    "n_epochs":1200,
    # "embed_dim":20,
    "loss":"standard",
    "scheduler":"restarts",
    "cosine_t":300,
    "lr":1e-4,
    "dropout":0.15,
    "batch_size":512,
    "shape":[512, 512, ],
    'subsample':100,
    # 'weight_decay':1e-4,
    # "unseen_frac":0., # no unseen class #will be overriden in experiment.py
}

def sample(params, const_params = {}):
    d = {}
    for p in params :
        d[p.name] = p.sample()
    d.update(const_params) #TODO : add a check for duplicate keys
    # additional processing
    if d['task'] != 'siamese':
        d['loss'] = 'standard'
    return d

async def worker(name, worker_name, gen, path, load_split = True, overwrite = False):
    log = open(f'logs/{name}/{worker_name}.log', 'w')
    for task_name, params, i in gen:
        # creating the command
    #python main.py ~/data/nabid_data/p53pilot/Data/RawData/filtered_feature_bc_matrix/wrangled/ test_nabid --dest-name nabid 

        cmd =  f'python main.py {path} {task_name} --dest-name {name} --verbose 1 '
        for arg, param in params.items():
                if isinstance(param, list):
                    param = ' '.join(map(str, param)) 
                cmd = cmd +f'--{arg} {param} ' 
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
           raise RuntimeError(f"""Error in task {task_name} on worker {worker_name} : exit code {exit_code}. 
                              See logs/{name}/{worker_name}.log for details.""")
    log.close()

async def main_f(args):
    #TODO : create directories for everything so that first workers don't try to do it at the same time
    #TODO : add a checkpoint to wait for first worker to split data before starting the others
    gen = ((args.name+f"_{i}", sample(PARAM_LIST, const_params=CONST_PARAMS), i) for i in range(args.i0, args.i0+args.n_models))
    workers = []
    for i in range(args.n_workers):
        workers.append(asyncio.create_task(worker(args.name, str(i), gen, args.path, overwrite=args.overwrite)))
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


    