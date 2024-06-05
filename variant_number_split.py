'''
Create splits for experiments assessing model capabilities on low number of variants
1. Split ~100 variants variants according in a 5-fold split stratified along synonymous/non-synonymous variants (5 splits)
2. Randomly subset 10 to 80 of the variants (12.5 to 100% of train variants), in 8 increments
3. Repeat previous step  10 times for statistics
'''
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
from os.path import join
from tqdm import tqdm

from main import get_paths, load_data

GENE = 'KRAS'
ROOT_DIR = f'models/variant_nb_{GENE}'
DATA_PATH = f'/home/lguirardel/data/perturb_comp/data/{GENE}/'
SEEN_FRAC = 0.1

if __name__ == '__main__':
    paths = get_paths(DATA_PATH, subset='processed')
    counts = load_data(*paths, group_wt_like= False, filt_variants=None,
                       standardize=False,)
    variants = counts.variant.unique()
    synonymous = [*map(lambda v : str(v)[0] == str(v)[-1], variants)]
    os.makedirs(ROOT_DIR, exist_ok=True)
    # split variants
    folds = StratifiedKFold(n_splits=5, shuffle=True).split(variants, synonymous)
    bars = tqdm(total=5*8*10)
    for i, (train_idx, test_idx) in enumerate(folds):
        test_variants = variants[test_idx]
        unseen_cells = pd.DataFrame(index = counts[counts.variant.isin(test_variants)].index).reset_index()
        for j in range(1,9):
            for k in range(10):
                bars.update(1)
                n_variants = int(len(train_idx) * j / 8)
                train_variants = np.random.choice(train_idx, n_variants, replace=False)
                train_variants = variants[train_variants] # drop non-valid, non-train variants
                seen_df = counts[counts.variant.isin(train_variants)]
                test_cells = seen_df.sample(frac=SEEN_FRAC).index
                train_cells = seen_df.index.difference(test_cells)
                # saving
                split_dir = join(ROOT_DIR, f'variant_nb_{GENE}_{i*8*10+(j-1)*10+k}/split')
                # print(split_dir)
                os.makedirs(split_dir, exist_ok=True)
                pd.DataFrame(index=train_cells).reset_index().to_csv(join(split_dir, 'index_0.csv'))
                pd.DataFrame(index=test_cells ).reset_index().to_csv(join(split_dir, 'index_1.csv'))
                unseen_cells.to_csv(join(split_dir, 'index_2.csv'))
                pd.Series(np.concatenate([train_variants, test_variants])).to_csv(join(split_dir, 'categories.csv'))

