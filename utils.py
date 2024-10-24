
from enum import Enum
from dataclasses import dataclass

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
