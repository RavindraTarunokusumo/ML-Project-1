from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Informative missingness constants ──────────────────────────────
# Columns where NaN means "feature absent", not "data missing"
INFORMATIVE_MISSING_CAT = [
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "Fence",
]
INFORMATIVE_MISSING_NUM = [
    "GarageYrBlt",
    "GarageArea",
    "GarageCars",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
]


class InformativeMissingFiller(BaseEstimator, TransformerMixin):
    """Fill NaN with 'None'/0 for features where missingness means absence."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in INFORMATIVE_MISSING_CAT:
            if col in X.columns:
                X[col] = X[col].fillna("None")
        for col in INFORMATIVE_MISSING_NUM:
            if col in X.columns:
                X[col] = X[col].fillna(0)
        return X


@dataclass(frozen=True)
class SplitConfig:
    val_size: float = 0.3
    test_size: float = 0.2
    random_state: int = 42
    shuffle: bool = True


def split_features_target(
    df: pd.DataFrame, target_col: str = "SalePrice"
) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series, cfg: SplitConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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
    
    if cfg.val_size == 0:
        # No validation set, return test only
        return (X_train, y_train), (pd.DataFrame(), pd.Series()), (X_holdout, y_holdout)
    
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
    X: pd.DataFrame,
    scale_numeric: bool = False,
    drop_missing_threshold: float | None = 0.8,
    drop_columns: Iterable[str] | None = ("Id",),
    use_pca: bool = False,
    pca_n_components: int | float | None = None,
) -> ColumnTransformer | Pipeline:
    drop_cols: set[str] = set()
    if drop_columns:
        drop_cols.update(col for col in drop_columns if col in X.columns)
    if drop_missing_threshold is not None:
        missing_pct = X.isna().mean()
        drop_cols.update(
            missing_pct[missing_pct > drop_missing_threshold].index.tolist()
        )

    X_filtered = X.drop(columns=sorted(drop_cols), errors="ignore")

    numeric_features = X_filtered.select_dtypes(include=["number"]).columns
    categorical_features = X_filtered.select_dtypes(exclude=["number"]).columns

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
    if not use_pca:
        return preprocessor

    pca_kwargs: dict[str, int | float] = {}
    if pca_n_components is not None:
        pca_kwargs["n_components"] = pca_n_components

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("pca", PCA(**pca_kwargs)),
        ]
    )