# Feature Engineering Report -- Ames Housing Dataset

## Current State

The pipeline currently engineers four features via `FeatureEngineer`:

| Feature | Formula |
| --- | --- |
| TotalSF | 1stFlrSF + 2ndFlrSF + TotalBsmtSF + GarageArea |
| TotalBath | FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath |
| HouseAge | YrSold - YearBuilt |
| RemodAge | YrSold - YearRemodAdd |

With all preprocessing flags enabled (`--fill_missing --ordinal --engineer --correct_skew --log_target`), XGBoost and CatBoost plateau around R^2 = 0.90-0.92 on the test set. The exploration report shows OverallQual (r=0.791) and GrLivArea (r=0.709) dominate, yet the models still leave ~8-10% of variance unexplained. The suggestions below target that residual error.

## Methodology

Each suggestion is derived from one of three sources:

1. **Domain knowledge** -- the Ames Assessor data dictionary (De Cock, 2011)
2. **Exploration report** -- correlation, ANOVA, skewness, and multicollinearity analysis
3. **Competitive analysis** -- top Kaggle kernels and published analyses of this dataset

Suggestions are prioritised by expected impact (High / Medium / Low) based on how strongly their source columns already correlate with SalePrice and how much new signal the transform is likely to add.

---

## 1. Interaction and Polynomial Features

### 1.1 OverallQual x GrLivArea (High)

The two strongest predictors interact non-linearly: a large house with low quality is worth less per sqft than a small house with excellent quality. Their product captures this.

```
QualArea = OverallQual * GrLivArea
```

Source columns: OverallQual (r=0.791), GrLivArea (r=0.709). This is the single highest-impact feature used in top Kaggle solutions.

### 1.2 OverallQual^2 (Medium)

SalePrice increases super-linearly with quality -- the jump from 8 to 10 is disproportionately larger than 4 to 6. A quadratic term captures the convex relationship.

```
QualSquared = OverallQual ** 2
```

### 1.3 GrLivArea x TotalBsmtSF (Medium)

Both measure livable space but in different dimensions (above-grade vs below-grade). Their interaction captures the value of usable total volume.

```
LivBsmtInteraction = GrLivArea * TotalBsmtSF
```

### 1.4 OverallQual x TotalSF (Medium)

Similar rationale to 1.1 but uses TotalSF (the existing engineered composite) instead of just above-grade area, giving a quality-weighted total footprint.

```
QualTotalSF = OverallQual * TotalSF
```

---

## 2. Area and Size Composites

### 2.1 TotalPorchSF (Medium)

Five separate porch columns fragment a single concept. Buyers care about total outdoor living space, not whether it is open, enclosed, or screened.

```
TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch + WoodDeckSF
```

Source columns individually have low correlation (r < 0.32), but their sum should correlate meaningfully with price in the upper tiers.

### 2.2 TotalFinBsmtSF (Medium)

The data splits basement into finished-type-1, finished-type-2, and unfinished. Finished basement space adds more value than unfinished.

```
TotalFinBsmtSF = BsmtFinSF1 + BsmtFinSF2
BsmtFinRatio = TotalFinBsmtSF / TotalBsmtSF   (where TotalBsmtSF > 0, else 0)
```

A high BsmtFinRatio signals a fully usable basement, which is a pricing premium over raw sqft.

### 2.3 HasMasVnr (Low)

MasVnrArea is highly skewed (skew=2.67) with 59% zeros. A binary indicator may be cleaner for tree splits.

```
HasMasVnr = 1 if MasVnrArea > 0 else 0
```

---

## 3. Temporal Features

### 3.1 WasRemodeled (Medium)

Distinguish houses that were never remodeled (YearRemodAdd == YearBuilt) from those that were. Remodeling resets perceived condition.

```
WasRemodeled = 1 if YearRemodAdd != YearBuilt else 0
```

### 3.2 YearsSinceGarage (Low)

GarageYrBlt (r=0.486 with SalePrice) captures garage age. Newer garages in old houses signal investment.

```
YearsSinceGarage = YrSold - GarageYrBlt   (where GarageYrBlt > 0, else -1)
```

### 3.3 SeasonSold (Low)

Real estate prices vary by season. MoSold alone has near-zero correlation (r=-0.029), but a seasonal grouping may capture the spring/summer premium.

```
SeasonSold = "Winter" if MoSold in [12,1,2]
             "Spring" if MoSold in [3,4,5]
             "Summer" if MoSold in [6,7,8]
             "Fall"   if MoSold in [9,10,11]
```

Encode as ordinal: Winter=0, Fall=1, Spring=2, Summer=3 (ordering by typical sale volume).

---

## 4. Boolean/Indicator Features

### 4.1 HasPool, HasGarage, HasBsmt, HasFireplace, Has2ndFlr (Medium)

Several features are dominated by zeros. Tree models already split on zero/non-zero but struggle when the feature is also highly skewed. Explicit binary flags help, especially after log transforms collapse the range.

```
HasPool      = 1 if PoolArea > 0 else 0
HasGarage    = 1 if GarageArea > 0 else 0
HasBsmt      = 1 if TotalBsmtSF > 0 else 0
HasFireplace = 1 if Fireplaces > 0 else 0
Has2ndFlr    = 1 if 2ndFlrSF > 0 else 0
```

### 4.2 IsNew (Medium)

Newly constructed homes (SaleCondition == "Partial" or YrSold == YearBuilt) have a distinct pricing model -- they often sell above typical comps.

```
IsNew = 1 if (YrSold == YearBuilt) else 0
```

---

## 5. Neighbourhood-Level Features

### 5.1 Neighbourhood Median Price (High)

Neighbourhood is the strongest categorical predictor (F=71.78, p~0). One-hot encoding with 25 levels is sparse. A target-encoded median price is far more informative for gradient-boosted trees.

**Implementation note:** Must be computed on the training fold only to avoid leakage. Use `TargetEncoder` from scikit-learn >= 1.3 or a manual grouped k-fold approach.

```
NeighMedianPrice = median(SalePrice) per Neighborhood    (fit on train only)
```

### 5.2 Neighbourhood Tier (Medium)

An alternative to 5.1 that avoids target leakage entirely: bin neighbourhoods into tiers (Low / MedLow / MedHigh / High) based on the training set median, then encode ordinally.

---

## 6. Quality Composite Scores

### 6.1 OverallScore (Medium)

OverallQual (r=0.791) and OverallCond (r=-0.078) are suspiciously uncorrelated. Their product captures houses that are both well-built and well-maintained.

```
OverallScore = OverallQual * OverallCond
```

### 6.2 ExterScore (Low)

Combine exterior quality and condition ordinal encodings into a single exterior score.

```
ExterScore = OrdinalEncode(ExterQual) * OrdinalEncode(ExterCond)
```

### 6.3 GarageScore (Low)

```
GarageScore = OrdinalEncode(GarageQual) * OrdinalEncode(GarageCond) * GarageCars
```

Combines quality, condition, and capacity into a single garage value proxy.

---

## 7. Ratio and Per-Unit Features

### 7.1 LotFrontageRatio (Low)

Lot frontage relative to area. Wide shallow lots vs narrow deep lots price differently.

```
LotFrontageRatio = LotFrontage / LotArea    (where LotArea > 0)
```

### 7.2 LivAreaPerRoom (Medium)

Average room size. Larger rooms command premiums in the higher market.

```
LivAreaPerRoom = GrLivArea / TotRmsAbvGrd    (where TotRmsAbvGrd > 0)
```

### 7.3 GarageAreaPerCar (Low)

Oversized garages (workshops, storage) vs tight garages.

```
GarageAreaPerCar = GarageArea / GarageCars    (where GarageCars > 0, else 0)
```

---

## 8. Feature Removal Candidates

Not all columns are useful. The exploration report flags several that should be considered for dropping to reduce noise:

| Feature | Reason |
| --- | --- |
| Utilities | 1459/1460 values are "AllPub" -- near-zero variance |
| Street | 1454/1460 "Pave" -- near-zero variance |
| PoolQC | 99.5% missing, only 7 non-null observations |
| MiscFeature | 96.3% missing |
| MiscVal | 96% zeros, skew=24.5, essentially a noise column |
| Condition2 | 1445/1460 "Norm", very low ANOVA F=2.72 |
| LandSlope | ANOVA F=1.96, p=0.14 -- not significant |
| Id | Row identifier, no predictive value (already dropped) |
| 3SsnPorch | 96.5% zeros, skew=10.3 |
| LowQualFinSF | 98.3% zeros, skew=9.0 |

---

## 9. MSSubClass Recoding

MSSubClass is coded as an integer (20, 30, 40, ...) but the data dictionary classifies it as **nominal** -- the numbers are building type codes, not a numeric scale. The pipeline currently treats it as numeric, which implies an ordinal relationship (30 > 20) that does not exist.

**Fix:** Cast MSSubClass to string before preprocessing so it enters the categorical pipeline branch and gets one-hot encoded instead.

---

## 10. LotFrontage Imputation Improvement

LotFrontage is 17.7% missing. The current pipeline uses median imputation across all rows, but lot frontage strongly correlates with neighbourhood (lots in the same subdivision tend to have similar frontage). Imputing by neighbourhood median is both domain-sound and empirically better:

```
LotFrontage_imputed = LotFrontage.fillna(
    df.groupby("Neighborhood")["LotFrontage"].transform("median")
)
```

---

## Prioritised Implementation Roadmap

Listed in order of expected impact on test R^2:

| Priority | Feature(s) | Impact | Complexity |
| --- | --- | --- | --- |
| 1 | QualArea (OverallQual x GrLivArea) | High | Low |
| 2 | NeighMedianPrice (target encoding) | High | Medium |
| 3 | MSSubClass recode to categorical | High | Low |
| 4 | QualSquared, QualTotalSF | Medium | Low |
| 5 | Binary flags (HasPool, HasGarage, etc.) | Medium | Low |
| 6 | TotalPorchSF, TotalFinBsmtSF, BsmtFinRatio | Medium | Low |
| 7 | LivAreaPerRoom, OverallScore | Medium | Low |
| 8 | WasRemodeled, IsNew | Medium | Low |
| 9 | Drop noise features (Utilities, Street, etc.) | Medium | Low |
| 10 | LotFrontage neighbourhood imputation | Low | Medium |
| 11 | SeasonSold, YearsSinceGarage | Low | Low |
| 12 | GarageScore, ExterScore, per-unit ratios | Low | Low |

---

## References

- De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. Journal of Statistics Education, 19(3). [Data documentation](https://jse.amstat.org/v19n3/decock/DataDocumentation.txt)
- [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- [NYC Data Science: House Price Prediction with Creative Feature Engineering](https://nycdatascience.com/blog/student-works/house-price-prediction-with-creative-feature-engineering-and-advanced-regression-techniques/)
- [Eamon Fleming: Predicting Housing Prices with Regression](https://www.eamonfleming.com/projects/housing-regression.html)
