# cv_setup.py
import sys
from sklearn.model_selection import KFold, StratifiedKFold
from validation.model_validation import CombPurgedKFoldCV
import pandas as pd

def initialize_cv_method(cv_method: str, X: pd.DataFrame, y: pd.Series, n_splits=5, n_test_splits=1, embargo_td=2):
    splits = []
    if cv_method == 'comb_purged_kfold':
        cv = CombPurgedKFoldCV(n_splits=n_splits, n_test_splits=n_test_splits, embargo_td=embargo_td)
        t1_ = X.index  # Assuming X is your feature DataFrame
        t1 = pd.Series(t1_).shift(100).fillna(0).astype(int)
        t2 = pd.Series(t1_).shift(-100).fillna(1e12).astype(int)
        splits = list(cv.split(X, pred_times=t1, eval_times=t2))
    elif cv_method == 'kfold':
        cv = KFold(n_splits=n_splits)
        splits = list(cv.split(X))
    elif cv_method == 'stratified_kfold':
        cv = StratifiedKFold(n_splits=n_splits)
        splits = list(cv.split(X, y))  # Assuming y is your target variable
    else:
        raise ValueError(f"Invalid cross-validation method: {cv_method}.")
    
    return splits



