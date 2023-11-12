import itertools as itt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    make_scorer,
)
from sklearn.model_selection import (
    TimeSeriesSplit,
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.pipeline import Pipeline


def time_series_split(X, n_splits, n_test_splits, n_purge, n_embargo):
    factorized_indices = np.unique(X["factorized"])

    # Calcola i confini dei fold
    fold_bounds = [
        (fold[0], fold[-1] + 1) for fold in np.array_split(factorized_indices, n_splits)
    ]

    # Crea la lista dei confini di test che diventeranno i set di test
    selected_fold_bounds = list(itt.combinations(fold_bounds, n_test_splits))

    # Inverte per iniziare i test dalla parte più recente del dataset
    selected_fold_bounds.reverse()

    for fold_bound_list in selected_fold_bounds:
        test_factorized_indices = np.empty(0)
        test_fold_bounds = []

        for fold_start, fold_end in fold_bound_list:
            # Registra i confini dell'attuale split di test
            if not test_fold_bounds or fold_start != test_fold_bounds[-1][-1]:
                test_fold_bounds.append((fold_start, fold_end))
            elif fold_start == test_fold_bounds[-1][-1]:
                test_fold_bounds[-1] = (test_fold_bounds[-1][0], fold_end)

            # Aggiunge gli indici al set di test
            test_factorized_indices = np.union1d(
                test_factorized_indices, factorized_indices[fold_start:fold_end]
            ).astype(int)

        # Calcola gli indici del set di addestramento
        train_indices = np.setdiff1d(factorized_indices, test_factorized_indices)

        # Applica il purging
        if n_purge > 0:
            purge_indices = np.arange(
                test_factorized_indices[0] - n_purge, test_factorized_indices[0]
            )
            train_indices = np.setdiff1d(train_indices, purge_indices)

        # Applica l'embargo
        if n_embargo > 0:
            embargo_indices = np.arange(
                test_factorized_indices[-1] + 1,
                test_factorized_indices[-1] + 1 + n_embargo,
            )
            train_indices = np.setdiff1d(train_indices, embargo_indices)

        yield train_indices, test_factorized_indices
