import numpy as np
def make_random_splits(df, train_frac=0.90, val_frac=0.05, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy(), df.iloc[test_idx].copy()
