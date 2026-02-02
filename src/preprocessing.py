from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class SplitConfig:
    val_size: float = 0.3
    test_size: float = 0.2
    random_state: int = 42
    shuffle: bool = True


def split_features_target(
    df: pd.DataFrame, target_col: str = "SalePrice"
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series, cfg: SplitConfig | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    cfg = cfg or SplitConfig()
    holdout_size = cfg.val_size + cfg.test_size
    
    # Split first into training and holdout (val + test)
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=holdout_size,
        random_state=cfg.random_state,
        shuffle=cfg.shuffle,
    )
    
    # Get relative test size
    test_ratio = cfg.test_size / holdout_size
    
    # Split holdout into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=test_ratio,
        random_state=cfg.random_state,
        shuffle=cfg.shuffle,
    )
    
    # Group and return
    train_df = (X_train, y_train)
    val_df = (X_val, y_val)
    test_df = (X_test, y_test)
    return train_df, val_df, test_df

def build_preprocessor(
    X: pd.DataFrame, scale_numeric: bool = False
) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns
    categorical_features = X.select_dtypes(exclude=["number"]).columns

    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), numeric_features),
            ("cat", Pipeline(steps=categorical_steps), categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor
