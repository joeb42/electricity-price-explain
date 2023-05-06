from typing import Callable
import numpy as np
import pandas as pd 
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from tqdm import tqdm

def preprocess(
        X: pd.DataFrame,
        imputer_func: Callable,
        scaler_func: Callable | None = None, 
        poly: Callable | None = None, 
        combine_country_features: bool = False
):
    X = pd.get_dummies(X)
    X.loc[:, X.isna().any()] = imputer_func(X.loc[:, X.isna().any()])
    assert not X.isna().any()
    if combine_country_features:
        features = ["CONSUMPTION", "GAS", "COAL", "HYDRO", "NUCLEAR", "SOLAR", "WINDPOW", "RESIDUAL_LOAD", "RAIN", "WIND", "TEMP"]
        for feature in features:
            X[f"DELTA_{feature}"] = X[f"DE_{feature}"] - X[f"FR_{feature}"]
            X[f"SUM_{feature}"] = X[f"DE_{feature}"] + X[f"FR_{feature}"]
    if scaler_func is not None:
        X = scaler_func(X)
    if poly is not None:
        X = poly(X) 
    return X

def train(
        model: BaseEstimator, 
        data: np.ndarray | pd.DataFrame, 
        n_splits: int, 
        imputer: TransformerMixin = SimpleImputer(), 
        scaler: TransformerMixin = StandardScaler(), 
        combine_country_features: bool = False,
        state: int = 42,
):
    X, y = data.drop(["TARGET", "ID", "DAY_ID"], axis=1), data["TARGET"]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=state)
    test_results, train_results = [], []
    for train_idx, test_idx in tqdm(kf.split(data)):
        X_train, y_train = X.copy().loc[train_idx], y[train_idx]
        X_test, y_test = X.copy().loc[test_idx], y[test_idx]
        X_train = preprocess(X_train, imputer.fit_transform, scaler.fit_transform, combine_country_features=combine_country_features)
        X_test = preprocess(X_test, imputer.transform, scaler.transform, combine_country_features=combine_country_features)
        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        train_rank = evaluate(train_preds, y_train)
        train_results.append(train_rank)
        test_preds = model.predict(X_test)
        test_rank = evaluate(test_preds, y_test)
        test_results.append(test_rank)
    return train_results, test_results

def evaluate(preds, y_test):
    rank = spearmanr(preds, y_test.values).correlation
    return rank