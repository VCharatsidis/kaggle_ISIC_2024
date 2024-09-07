import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

import lightgbm as lgb
import catboost as cb
import xgboost as xgb

from sklearn.utils import resample

import optuna

from public_utils import custom_metric


def lgb_objective(trial, sampling_ratio, df_train, feature_cols, target_col, group_col):
    params = {
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth':         trial.suggest_int('max_depth', 3, 8),

        'objective': 'binary',
        'verbosity': -1,
        'num_iterations': 200,
        'boosting_type': 'gbdt',
        'lambda_l1': 0.08758718919397321,
        'lambda_l2': 0.0039689175176025465,

        'num_leaves': 103,
        'colsample_bytree': 0.8329551585827726,
        'colsample_bynode': 0.4025961355653304,
        'bagging_fraction': 0.7738954452473223,
        'bagging_freq': 4,
        'min_data_in_leaf': 85,
        'scale_pos_weight': 2.7984184778875543,
    }

    estimator = Pipeline([
        ('sampler_1', RandomOverSampler(sampling_strategy=0.003)),
        ('sampler_2', RandomUnderSampler(sampling_strategy=sampling_ratio)),
        ('classifier', lgb.LGBMClassifier(**params)),
    ])

    X = df_train[feature_cols]
    y = df_train[target_col]
    groups = df_train[group_col]
    cv = StratifiedGroupKFold(5, shuffle=True)

    val_score = cross_val_score(
        estimator=estimator,
        X=X, y=y,
        cv=cv,
        groups=groups,
        scoring=custom_metric,
    )

    return np.mean(val_score)


def cb_objective(trial, seed, sampling_ratio, df_train, feature_cols, target_col, group_col, cat_cols):
    params = {
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth':         trial.suggest_int('max_depth', 4, 8),

        'loss_function': 'Logloss',
        'iterations': 200,
        'verbose': False,
        'random_state': seed,
        'scale_pos_weight': 2.6149345838209532,
        'l2_leaf_reg': 6.216113851699493,
        'subsample': 0.6249261779711819,
        'min_data_in_leaf': 24,
        'cat_features': cat_cols,
    }

    estimator = Pipeline([
        ('sampler_1', RandomOverSampler(sampling_strategy=0.003, random_state=seed)),
        ('sampler_2', RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=seed)),
        ('classifier', cb.CatBoostClassifier(**params)),
    ])

    X = df_train[feature_cols]
    y = df_train[target_col]
    groups = df_train[group_col]
    cv = StratifiedGroupKFold(5, shuffle=True)

    val_score = cross_val_score(
        estimator=estimator,
        X=X, y=y,
        cv=cv,
        groups=groups,
        scoring=custom_metric,
    )

    return np.mean(val_score)


def xgb_objective(trial, seed, sampling_ratio, df_train, feature_cols, target_col, group_col):
    params = {
        'objective':          'binary:logistic',
        'learning_rate':      trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth':          trial.suggest_int('max_depth', 4, 8),

        'enable_categorical': True,
        'tree_method': 'hist',
        'random_state': seed,
        'lambda': 8.879624125465703,
        'alpha': 0.6779926606782505,
        'subsample': 0.6012681388711075,
        'colsample_bytree': 0.8437772277074493,
        'colsample_bylevel': 0.5476090898823716,
        'colsample_bynode': 0.9928601203635129,
        'scale_pos_weight': 3.29440313334688,
    }

    estimator = Pipeline([
        ('sampler_1', RandomOverSampler(sampling_strategy=0.003, random_state=seed)),
        ('sampler_2', RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=seed)),
        ('classifier', xgb.XGBClassifier(**params)),
    ])

    X = df_train[feature_cols]
    y = df_train[target_col]
    groups = df_train[group_col]
    cv = StratifiedGroupKFold(5, shuffle=True)

    val_score = cross_val_score(
        estimator=estimator,
        X=X, y=y,
        cv=cv,
        groups=groups,
        scoring=custom_metric,
    )

    return np.mean(val_score)