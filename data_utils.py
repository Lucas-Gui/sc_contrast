import gzip
import pandas as pd
import numpy as np

GENE = 'TP53'


def load_data(mtx_path, gene_path, cell_path, v2c_path, variant_path)-> pd.DataFrame:
    '''
    *_path : paths to gzipped mtx or csv files
    except variant_path, which refers to an excel file
    '''
    with gzip.open(mtx_path) as file: #read counts
        counts = pd.read_csv(file, skiprows=3,header=None, sep=' ')
    counts.columns = ['cell','gene','reads']
    counts[['cell','gene']] -= 1
    with gzip.open(gene_path) as file: # read gene names
        genes = pd.read_csv(file, header=None, sep=' ', ).squeeze()
    genes.name = 'gene'
    with gzip.open(cell_path) as file:
        cells = pd.read_csv(file,header=None, sep=' ', ).squeeze()
    cells.name = 'cell'
    #pivot
    counts['gene_name'] = genes.loc[counts.gene].reset_index(drop=True)
    counts = counts.pivot(index='cell',columns='gene_name',values='reads')

    with gzip.open(v2c_path) as file: # read cell tags
        v2c = pd.read_csv(file, sep='\t', )
    v2c = v2c[['cell','variant']]
    # read variant impact class
    variant_data = pd.read_excel(variant_path, sheet_name=GENE, index_col='Variant')
    variant_data = variant_data['Variant functional class']
    variant_data = variant_data.replace({'Impactful IV (gain-of-function)':'Impactful IV'})
    variant_data = variant_data[variant_data!='unavailable']
    # v2c = v2c[v2c['variant'].isin(variant_class.index)]
    v2c = v2c.merge(variant_data, how='left', left_on='variant', right_index=True ).dropna(axis=0)
    #impute variant/impact class to cell and drop cells with missing values
    counts = counts.merge(v2c, how='left', left_index=True, right_on='cell').dropna(axis=0)
    return counts

def split(data:pd.DataFrame, x_cell = 0.25, x_var = 0.25): #TODO : groupby sample ?
    '''
    First remove a x_var fraction of variants a unseen-class test set, 
    and then a x_cell fraction as a seen-class test set 
    '''
    test_vars = np.random.choice(data['variant'].unique(), p=x_var, replace=False)
    test_unseen = data[data['variant'].isin(test_vars)]
    train = data[~data['variant'].isin(test_vars)]
    test_seen = train.sample(frac=x_cell, replace=False)
    train = train[train.index.difference(test_seen.index)]

    return train, test_seen, test_unseen


    
    
