from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

# ── Ordinal encoding constants ─────────────────────────────────────
# "None" is prepended so that absent features (from InformativeMissingFiller)
# get the lowest rank.  OrdinalEncoder maps index → value, so None=0, Po=1…
_QUALITY_SCALE = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
QUALITY_COLS = [
    "ExterQual",
    "ExterCond",
    "KitchenQual",
    "BsmtQual",
    "BsmtCond",
    "HeatingQC",
    "FireplaceQu",
    "GarageQual",
    "GarageCond",
]

# Other ordinal features — each maps low → high
ORDINAL_MAPPINGS: dict[str, list[str]] = {
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
    "PavedDrive": ["N", "P", "Y"],
    "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
    "LotShape": ["IR3", "IR2", "IR1", "Reg"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "Utilities": ["NoSeWa", "AllPub"],
    "CentralAir": ["N", "Y"],
}

# Union of all ordinal column names
ALL_ORDINAL_COLS = set(QUALITY_COLS) | set(ORDINAL_MAPPINGS)

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


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create composite features when source columns are present."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if {"1stFlrSF", "2ndFlrSF", "TotalBsmtSF", "GarageArea"}.issubset(X.columns):
            X["TotalSF"] = X["1stFlrSF"] + X["2ndFlrSF"] + X["TotalBsmtSF"] + X["GarageArea"]
        if {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}.issubset(
            X.columns
        ):
            X["TotalBath"] = (
                X["FullBath"]
                + 0.5 * X["HalfBath"]
                + X["BsmtFullBath"]
                + 0.5 * X["BsmtHalfBath"]
            )
        if "YearBuilt" in X.columns and "YrSold" in X.columns:
            X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
        if "YearRemodAdd" in X.columns and "YrSold" in X.columns:
            X["RemodAge"] = X["YrSold"] - X["YearRemodAdd"]
        if {"OverallQual", "GrLivArea"}.issubset(X.columns):
            X["QualArea"] = X["OverallQual"] * X["GrLivArea"]
        if "MSSubClass" in X.columns:
            X["MSSubClass"] = X["MSSubClass"].astype(str)
        return X


class NeighborhoodTargetEncoder(BaseEstimator, TransformerMixin):
    """Target encode Neighborhood using a summary statistic (median or mean)."""

    def __init__(self, column: str = "Neighborhood", stat: str = "median"):
        self.column = column
        self.stat = stat
        self.mapping_ = None
        self.global_stat_ = None

    def fit(self, X, y=None):
        if y is None:
            return self
        
        # Ensure X is a DataFrame and y is a Series
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
        # We need to align indexes if they are different
        y_series.index = X.index
        
        if self.stat == "median":
            self.mapping_ = y_series.groupby(X[self.column]).median()
            self.global_stat_ = y_series.median()
        else:
            self.mapping_ = y_series.groupby(X[self.column]).mean()
            self.global_stat_ = y_series.mean()
        return self

    def transform(self, X):
        if self.mapping_ is None:
            return X
        X = X.copy()
        col_name = "NeighMedianPrice" if self.stat == "median" else "NeighMeanPrice"
        X[col_name] = X[self.column].map(self.mapping_)
        X[col_name] = X[col_name].fillna(self.global_stat_)
        return X


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


class SkewnessCorrector(BaseEstimator, TransformerMixin):
    """Apply log1p to numeric columns with absolute skewness above threshold."""

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            skew = X.skew(numeric_only=True)
        else:
            skew = pd.DataFrame(X).skew()
        self.skewed_cols_ = skew[skew.abs() > self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(X, pd.DataFrame):
            for col in self.skewed_cols_:
                if col in X.columns:
                    X[col] = np.log1p(X[col].clip(lower=0))
        else:
            X = pd.DataFrame(X)
            for idx in self.skewed_cols_:
                if idx < X.shape[1]:
                    X.iloc[:, idx] = np.log1p(X.iloc[:, idx].clip(lower=0))
            X = X.values
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
    X: pd.DataFrame, y: pd.Series, 
    cfg: SplitConfig | None = None
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

def _ordinal_categories_for(col: str) -> list[str]:
    """Return the ordered category list for an ordinal column."""
    if col in ORDINAL_MAPPINGS:
        return ORDINAL_MAPPINGS[col]
    # Quality columns share the same scale
    return _QUALITY_SCALE


def build_preprocessor(
    X: pd.DataFrame,
    scale_numeric: bool = False,
    drop_missing_threshold: float | None = 0.8,
    drop_columns: Iterable[str] | None = ("Id",),
    use_pca: bool = False,
    pca_n_components: int | float | None = None,
    use_ordinal_encoding: bool = False,
    correct_skewness: bool = False,
) -> ColumnTransformer | Pipeline:
    drop_cols: set[str] = set()
    if drop_columns:
        drop_cols.update(
            col for col in drop_columns if col in X.columns
        )
    if drop_missing_threshold is not None:
        missing_pct = X.isna().mean()
        drop_cols.update(
            missing_pct[
                missing_pct > drop_missing_threshold
            ].index.tolist()
        )

    X_filtered = X.drop(
        columns=sorted(drop_cols), errors="ignore"
    )

    numeric_features = X_filtered.select_dtypes(
        include=["number"]
    ).columns
    categorical_features = X_filtered.select_dtypes(
        exclude=["number"]
    ).columns

    # ── Numeric branch ─────────────────────────────────────────
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if correct_skewness:
        numeric_steps.append(("skew_fix", SkewnessCorrector()))
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    # ── Categorical branch (one-hot) ──────────────────────────
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            ),
        ),
    ]

    # ── Ordinal branch (optional) ─────────────────────────────
    if use_ordinal_encoding:
        ordinal_cols = [
            c
            for c in categorical_features
            if c in ALL_ORDINAL_COLS
        ]
        remaining_cat = [
            c
            for c in categorical_features
            if c not in ALL_ORDINAL_COLS
        ]

        ordinal_categories = [
            _ordinal_categories_for(c) for c in ordinal_cols
        ]
        ordinal_steps = [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    else:
        ordinal_cols = []
        remaining_cat = list(categorical_features)

    # ── Assemble ColumnTransformer ────────────────────────────
    transformers = [
        ("num", Pipeline(steps=numeric_steps), numeric_features),
        (
            "cat",
            Pipeline(steps=categorical_steps),
            remaining_cat,
        ),
    ]
    if ordinal_cols:
        transformers.append(
            (
                "ord",
                Pipeline(steps=ordinal_steps),
                ordinal_cols,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
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