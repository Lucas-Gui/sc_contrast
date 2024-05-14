'''Run models in parallel with shared data'''
from argparse import ArgumentParser
import torch
import torch.multiprocessing as mp # allow tensor sharing
from torch import Tensor

from hyperparam import NumParamSampler, CatParamSampler, ShapeSampler

## CHANGE HERE TO CHANGE EXPERIMENT



## MULTIPROCESSING CODE



## MAIN
if __name__ == '__main__':
    parser = ArgumentParser()

    args = parser.parse_args()