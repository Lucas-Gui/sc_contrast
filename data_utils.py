import gzip
import pandas as pd


def load_data(mtx_path, gene_path, cell_path, v2c_path)-> pd.DataFrame:
    '''
    *_path : paths to gzipped mtx or csv files
    '''
    with gzip.open(mtx_path) as file:
        counts = pd.read_csv(file, skiprows=3,header=None, sep=' ')
    counts.columns = ['cell','gene','reads']
    counts[['cell','gene']] -= 1
    with gzip.open(gene_path) as file:
        genes = pd.read_csv(file, header=None, sep=' ', ).squeeze()
    genes.name = 'gene'
    with gzip.open(cell_path) as file:
        cells = pd.read_csv(file,header=None, sep=' ', ).squeeze()
    cells.name = 'cell'
    counts['gene_name'] = genes.loc[counts.gene].reset_index(drop=True)
    counts = counts.pivot(index='cell',columns='gene_name',values='reads')
    with gzip.open(v2c_path) as file:
        v2c = pd.read_csv(file, sep='\t', )
    v2c = v2c['cell','variant']
    counts = counts.merge(v2c, how='left', left_index=True, right_on='cell')
    
