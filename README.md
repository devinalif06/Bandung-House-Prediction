# House Price Prediction — Greater Bandung

A regression model to predict house prices in Greater Bandung (Bandung City, Bandung Regency, and West Bandung Regency) based on property characteristics such as land area, building area, number of rooms, and location.

## Summary

| | |
|---|---|
| **Problem type** | Regression |
| **Target** | `Price` (house price, IDR) |
| **Dataset size** | 32,536 rows, 0 missing values, 0 duplicates |
| **Final model** | Extra Trees Regressor (tuned) |
| **RMSE (test set)** | ± IDR 635 million |
| **MAE (test set)** | ± IDR 349 million |
| **MAPE (test set)** | 19.11% |
| **R² (test set)** | 0.8814 |

## Dataset

The dataset contains house listings in Greater Bandung with the following features:

| Column | Description |
|---|---|
| `Price` | House price (target) |
| `Location` | Sub-district/area name (66 unique categories) |
| `City/Regency` | City/Regency (3 categories: Bandung City, Bandung Regency, West Bandung Regency) |
| `Bedroom`, `Bathroom` | Number of bedrooms & bathrooms |
| `Carport` | Number of carports |
| `Land`, `Building` | Land area & building area (m²) |
| `Month` | Listing month (dropped — weak correlation with target) |
| `Latitude`, `Longitude` | Location coordinates (dropped — information already captured by `Location`) |

## Methodology

The workflow follows a standard data science pipeline, from raw data to a deployable model:

1. **Initial Inspection** — check data structure, missing values, duplicates, cardinality of categorical columns
2. **Exploratory Data Analysis** — target distribution (right-skewed → `log1p`), numerical feature distributions, outlier detection (IQR), correlation with target, ANOVA test for categorical features, multicollinearity check (VIF)
3. **Data Preprocessing** — train-test split (80:20) performed **before** any transformation to prevent data leakage
4. **Feature Engineering** — derived features created (`Bath_per_Bed_Ratio`, `Building_per_Bedroom`, `Land_Sisa`, `Building_Coverage_Ratio`, `Total_Facility`), then **tested via ablation**: 14 models compared with vs. without these derived features using 5-fold cross-validation, to confirm the engineered features actually add value rather than just added complexity
5. **Modeling** — 14 models compared (baseline, linear/regularized, tree-based, gradient boosting), evaluated with 5-fold cross-validation on both log scale and original (Rupiah) scale
6. **Hyperparameter Tuning** — `RandomizedSearchCV` (30 combinations, 5-fold) on the best-performing model
7. **Evaluation & Error Analysis** — residual plots, learning curve, final evaluation on a held-out test set never seen during model selection or tuning
8. **Feature Importance** — analysis of each feature's contribution to the predictions

## Model Comparison Results

Top 5 models by performance (5-fold CV, original Rupiah scale):

| Model | RMSE (IDR) | MAE (IDR) | MAPE | R² |
|---|---|---|---|---|
| **Extra Trees** | 670M | 359M | 24.21% | 0.8741 |
| Random Forest | 677M | 382M | 25.02% | 0.8714 |
| XGBoost | 800M | 497M | 30.63% | 0.8206 |
| HistGradientBoosting | 827M | 536M | 30.63% | 0.8081 |
| CatBoost | 829M | 527M | 31.45% | 0.8072 |

**Extra Trees Regressor** was selected as the final model. Notably, the ablation test showed this model actually performed better **without** the additional derived features — the engineered features introduced substantial multicollinearity (VIF > 25 on some features) without a proportional performance gain for this particular model.

## Key Insights

- **Location has a significant effect** on price (ANOVA, p < 0.05) at both the `Location` and `City/Regency` level
- **Land area and building area** are the largest contributors to price prediction, dominating the feature importance results
- The model generalizes well — the small gap between cross-validation and test set scores indicates **no significant overfitting**
- A MAPE of ~19% suggests there is still room for improvement, particularly for high-end properties, which are relatively underrepresented in the dataset

## Project Structure

```
├── data/
│   └── clean_df.csv
├── models/
│   └── house_price_pipeline.pkl
├── notebooks/
│   └── house_price_prediction.ipynb
├── requirements.txt
└── README.md
```

## Installation & Usage

```bash
git clone <repo-url>
cd house-price-prediction
pip install -r requirements.txt
jupyter notebook notebooks/house_price_prediction.ipynb
```

## Using the Saved Model

```python
import joblib
import numpy as np
import pandas as pd

model = joblib.load("models/house_price_pipeline.pkl")

new_data = pd.DataFrame({
    "Location": ["Buah Batu"],
    "City/Regency": ["Bandung City"],
    "Bedroom": [3],
    "Bathroom": [2],
    "Carport": [1],
    "Land": [90],
    "Building": [70],
})

pred_log = model.predict(new_data)
predicted_price = np.expm1(pred_log)
print(f"Estimated price: IDR {predicted_price[0]:,.0f}")
```

## Limitations & Future Work

- The dataset lacks information on building condition, year built, or proximity to public facilities (schools, hospitals, highway access) — adding such external features could further improve accuracy
- Model performance degrades for high-end properties, as they are relatively underrepresented in the dataset
- Further validation on data from a different time period (out-of-time validation) would provide a better picture of the model's stability over time

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `LightGBM` · `CatBoost` · `statsmodels` · `matplotlib` · `seaborn`

## Contact

Built by Devin Alif — Information Systems student, Institut Teknologi Sepuluh Nopember (ITS)
