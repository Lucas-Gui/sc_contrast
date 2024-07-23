import gzip
import pandas as pd
import numpy as np
from contrastive_data import Data
from os.path import join
import os
from enum import Enum
from typing import *

from psutil import Process

EPS_STD = 1e-3

def filter_nabid_data(counts:pd.DataFrame, gene_thr = 200, mito_freq_thr = 0.075, reads_thr = 180,
                      log_mito_counts: float|None = 2.53): 
    '''
    gene_thr : minimum number of genes expressed in a cell
    mito_freq_thr : maximum fraction of mitochondrial reads
    reads_thr : minimum number of reads
    log_mito_counts : optional. Minimum log10 of the number of mitochondrial reads
    '''
    n0 = len(counts)
    counts = counts[(counts>0).sum(axis=1) > gene_thr] 
    mt = counts.columns.str.startswith('MT-')
    counts = counts[counts.loc[:,mt].sum(axis=1) < mito_freq_thr*counts.sum(axis=1)]
    counts = counts[counts.sum(axis=1) > reads_thr]
    if log_mito_counts is not None:
        counts = counts[counts.loc[:,mt].sum(axis=1) > 10**log_mito_counts]
    # remove all-0 genes
    counts = counts.loc[:,counts.sum(axis=0) > 0]
    print(f"\t\t{n0-len(counts)}/{n0} cells removed.", flush=True)
    return counts.copy()

def load_data(mtx_path, gene_path, cell_path, v2c_path, variant_path=None,
              cell_meta_path=None, group_wt_like=False, filt_variants = None,
              standardize=False, filt_cells=False, filter_kwargs={}
              )-> pd.DataFrame:
    '''
    *_path : paths to gzipped mtx or csv files
    except variant_path, which refers to an unzipped csv file
    filter_kwargs : arguments to pass to filter_nabid_data
    '''
    print('\tReading files...', flush=True)
    print('\t\tReading matrix', flush=True)
    genes = pd.read_csv(gene_path, header=None, sep=' ', compression='gzip').squeeze()
    cells = pd.read_csv(cell_path, header=None, sep=' ', compression='gzip').squeeze()
    print('\t\tReading genes and cells', flush=True)
    with gzip.open(mtx_path) as file: #read counts
        counts = pd.read_csv(file, skiprows=3,header=None, sep=' ')
    counts.columns = ['cell','gene','reads'] 
    counts[['cell','gene']] -= 1 #file is imported from 1-indexed array format
    check_mtx_columns(counts, genes, cells) # P-Seq is CELL x GENE, GFP is GENE x CELL
    print('\t\tReading variant data', flush=True)
    v2c = pd.read_csv(v2c_path, sep='\t', usecols=['cell','variant'], compression='gzip')

    if variant_path is not None:
        variant_data = pd.read_csv(variant_path, index_col=0)
        variant_data = variant_data['Variant functional class']
        # read variant impact class
        variant_data = variant_data.replace({'Impactful IV (gain-of-function)':'Impactful IV'})
        variant_data = variant_data[variant_data!='unavailable']
        v2c = v2c.merge(variant_data, how='left', left_on='variant', right_index=True ).dropna(axis=0)
    else:
        v2c['Variant functional class'] = 'Unassigned'

    print('\tMerging and processing...' ,flush=True)

    #pivot 
    counts['gene_name'] = genes.loc[counts.gene].reset_index(drop=True)
    counts['cell_name'] = cells.loc[counts.cell].reset_index(drop=True)
    # counts = counts.drop(columns=['cell','gene']) #unnecessary : included in pivot
    counts = counts.pivot(index='cell_name',columns='gene_name',values='reads')
    counts = counts.fillna(0) # pivot will create NA where the .mtx file was sparse
    if filt_cells:
        print('\t\tFiltering cells and removing variants subsequently absent.')
        print(f'\t\t{Process().memory_info().rss/1024**3:.2f} GB used')
        counts = filter_nabid_data(counts, **filter_kwargs)
    if cell_meta_path is not None:
    # read cell metadata (for cell cycle)
        cell_meta = pd.read_csv(file, index_col=-1, compression='gzip')
        counts['cycle'] = cell_meta['phase.multi'].replace('G0', 'Uncycling')
    else:
        counts['cycle'] = 'Unassigned'
    counts['cycle'] = counts['cycle'].astype('category')

    # v2c = v2c[v2c['variant'].isin(variant_class.index)]
    print('\t\tMerging counts')
    #impute variant/impact class to cell and drop cells with missing values
    counts = counts.merge(v2c, how='left', left_index=True, right_on='cell').dropna(axis=0).set_index('cell')
    counts = counts[counts.variant != 'unassigned']
    print(f'\t\t{Process().memory_info().rss/1024**3:.2f} GB used', flush=True)
    if filt_variants is not None:
        counts = counts[counts.variant.isin(filt_variants)]
    if group_wt_like:
        print('\t\tGrouping control variants.')
        filt = (counts['variant'].str[0] == counts['variant'].str[-1])  | (counts.variant == 'WT')
        print(f"\t\tCasting {counts[filt].variant.nunique()} variants to 'control' ")
        counts.loc[filt, 'variant']= 'control'
    if standardize:
        print('\t\tRemoving low variance genes and standardizing data')
        low_var_genes = counts.iloc[:,:-3].std()< EPS_STD
        print('\t\t',(low_var_genes).sum(), f"genes with std < {EPS_STD:.1e} dropped out of {len(counts.columns)-3}")
        counts = counts.drop(columns=low_var_genes[low_var_genes].index)
        counts.iloc[:,:-3] = (counts.iloc[:,:-3] - counts.iloc[:,:-3].mean())/counts.iloc[:,:-3].std()

    print(f"\t\t{len(counts['variant'].unique())} variant classes")
    counts['variant'] = counts['variant'].astype('category')
    return counts


def check_mtx_columns(counts:pd.DataFrame, genes:pd.Series, cells:pd.Series):
    '''Check that the columns in the mtx file are consistent with the gene and cell files and permute columns if necessary'''
    n1, n2 = counts.gene.max(), counts.cell.max()
    m1, m2 = genes.shape[0]-1, cells.shape[0]-1
    if n1 == m2 and n2 == m1:
        print('\t'*3+'Permuting gene columns to match gene file') 
        if n1 == n2:
            print("Number of genes and cells are equal. Please check that the mtx file is (genes x cells) and not (cells x genes).")
        counts[['gene','cell']] = counts[['cell','gene']]
    elif n2 != m2 or n1 != m1:
        print('counts')
        print(counts.shape)
        print(counts.head())
        print('genes')
        print(genes.shape)
        print(genes.head())
        print(counts.gene.max())
        print('cells')
        print(cells.shape)
        print(cells.head())
        print(counts.cell.max())
        raise ValueError('Gene file does not match mtx file')

def split_dataframes(data:pd.DataFrame, x_cell = 0.25, x_var = 0.25): #TODO : groupby sample ?
    '''
    First remove a x_var fraction of variants a unseen-class test set, 
    and then a x_cell fraction as a seen-class test set.
    Keep 'control' variant in training set.
    Return three DataFrames for train, test seen and test unseen
    '''
    variants = data.loc[data.variant != 'control', 'variant'].unique() 
    # reorder categories such that codes 0 ... m-1 are in seen and m ... n-1 in unseen
    variants = np.random.permutation(variants)
    if 'control' in data.variant.cat.categories:
        variants = np.concatenate((np.array(['control']), variants))
    data['variant'] = data.variant.cat.reorder_categories(variants)
    if x_var == 0:
        test_vars = []
        test_unseen = pd.DataFrame(columns=data.columns)
    else:
        test_vars = variants[-int(len(variants)*x_var):]
        test_unseen = data[data['variant'].isin(test_vars)]
    train = data[~data['variant'].isin(test_vars)]
    test_seen = train.sample(frac=x_cell, replace=False)
    train = train.loc[train.index.difference(test_seen.index)] #also sort cells in train by tag lexical order, but that's okay
    print(f"Train length: {len(train)}")
    print(f"Seen test length  : {len(test_seen)}")
    print(f"Unseen test length: {len(test_unseen)}")
    print(f"Categories in seen : {train.variant.nunique()}")
    print(f"Categories in unseen : {test_unseen.variant.nunique()}")
 
    return train, test_seen, test_unseen

def split(data:Data, x_cell=0.25, x_var=0.25):
    '''
    First remove a x_var fraction of variants a unseen-class test set, 
    and then a x_cell fraction as a seen-class test set.
    Keep 'control' variant in training set.
    Return three Data objects for train, test seen and test unseen
    '''
    variants = data.variants[data.variants != 'control'].unique()
    # reorder categories such that codes 0 ... m-1 are in seen and m ... n-1 in unseen
    variants = np.random.permutation(variants)
    if 'control' in data.variants.cat.categories:
        variants = np.concatenate((np.array(['control']), variants))
    data.variants = data.variants.cat.reorder_categories(variants)
    if x_var == 0:
        test_vars = []
        unseen = None #make an empty Data instead ?
        filt_unseen = pd.Series(False, index=data.variants.index)
    else:
        test_vars = variants[-int(len(variants)*x_var):]
        filt_unseen = data.variants.isin(test_vars)
        unseen = data.subset(filt_unseen)
    # randomly select x_cell fraction of cells for seen test set
    filt_seen = pd.Series(
        np.random.rand(len(data.variants)) < x_cell, 
        index=data.variants.index
        )
    # remove cells in seen and train from unseen
    filt_train = ~filt_seen & ~filt_unseen
    filt_seen = filt_seen & ~filt_unseen
    seen = data.subset(filt_seen)
    train = data.subset(filt_train)
    print(f"Train length: {len(train)}")
    print(f"Seen test length  : {len(seen)}")
    if unseen is not None:
        print(f"Unseen test length: {len(unseen)}")
    else :
        print("No unseen test set") 
    return train, seen, unseen
