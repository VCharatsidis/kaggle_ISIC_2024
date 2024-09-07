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


def lgb_objective(trial, sampling_ratio, df_train, feature_cols, target_col, group_col, seed):
    params = {
        'objective':         'binary',
        'verbosity':         -1,
        'num_iterations': 200,
        'boosting_type':    'gbdt',
        # 'lambda_l1':         trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        # 'lambda_l2':         trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        # 'learning_rate':    0.05536251727552012,
        # 'max_depth':        5,
        # 'num_leaves':        trial.suggest_int('num_leaves', 16, 256),
        # 'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.4, 1.0),
        # 'colsample_bynode':  trial.suggest_float('colsample_bynode', 0.4, 1.0),
        #  'bagging_fraction': 0.8366733523272176,
        # 'bagging_freq':      trial.suggest_int('bagging_freq', 1, 7),
        # 'min_data_in_leaf':  trial.suggest_int('min_data_in_leaf', 5, 100),
        # 'scale_pos_weight' : trial.suggest_float('scale_pos_weight', 0.8, 4.0),

        'learning_rate': trial.suggest_float('learning_rate', 1e-2, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 8),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),

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
        'loss_function':     'Logloss',
        'iterations':        200,
        'verbose':           False,
        'random_state':      seed,
        'learning_rate':     trial.suggest_float('learning_rate', 1e-2, 1e-1, log=True),
        'max_depth':         trial.suggest_int('max_depth', 5, 8),
        # 'l2_leaf_reg':       trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'subsample':         trial.suggest_float('subsample', 0.4, 1.0),
        # 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
        # 'min_data_in_leaf':  trial.suggest_int('min_data_in_leaf', 5, 100),
        # 'scale_pos_weight':  trial.suggest_float('scale_pos_weight', 0.8, 4.0),
        #'bootstrap_type':    'Bayesian',  # Optional: depending on your use case, you may want to tune this as well
        'cat_features': cat_cols,

    }

    estimator = Pipeline([
        ('sampler', RandomUnderSampler(sampling_strategy=sampling_ratio)),
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
        'n_estimators':       200,
        'tree_method':        'hist',
        'random_state':       seed,
        # 'learning_rate': 0.05056902007063551,
        # 'max_depth': 8,
        # 'subsample': 0.744401997795449,
        # 'lambda':             trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        # 'alpha':              trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        # 'colsample_bytree':   trial.suggest_float('colsample_bytree', 0.4, 1.0),
        # 'colsample_bynode':   trial.suggest_float('colsample_bynode', 0.4, 1.0),
        # 'scale_pos_weight':   trial.suggest_float('scale_pos_weight', 0.8, 4.0),
        'enable_categorical': True,

        'learning_rate': trial.suggest_float('learning_rate', 1e-2, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 6, 8),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
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