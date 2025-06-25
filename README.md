# Descriptive Analytics of Retail Car Seat Sales: A Regression-Based Exploration

## Overview

This project conducts a **descriptive analytics** investigation into the `Carseats` dataset using regression models. The primary objective is to **understand the key drivers of child car seat sales** across 300 retail locations using linear and ensemble machine learning models. Rather than focusing on predicting future outcomes, this analysis emphasizes interpreting model coefficients and understanding feature importance—offering business insight into how different variables affect sales.

## Dataset

**Source**: `ISLP` package  
**Observations Used**: First 300 entries of the full dataset  
**Target Variable**: `Sales` (unit sales in thousands)  

### Features:

- `CompPrice`: Competitor's price
- `Income`: Average community income
- `Advertising`: Local ad spend
- `Population`: Local population
- `Price`: Company's retail price
- `ShelveLoc`: Shelf location quality (`Bad`, `Medium`, `Good`)
- `Age`: Average population age
- `Education`: Education level
- `Urban`: Urban or rural (`Yes`/`No`)
- `US`: Whether the store is in the U.S.

## Methodology

### 1. **Descriptive Analysis & EDA (Exploratory Data Analysis)**
- Checked for nulls, duplicates, and infinite values
- Visualized distributions and relationships using histograms and bar plots
- Explored summary statistics for sales and predictors

### 2. **Data Preprocessing**
- Used `RobustScaler` on numerical features
- Applied `OneHotEncoder` to categorical variables
- Performed an 80/20 train-test split

### 3. **Modeling With K-Fold Cross-Validation (`k=5`)**

| Model         | Purpose |
|---------------|---------|
| Linear        | Interpret coefficient signs/magnitudes |
| Ridge         | Shrink coefficients to reduce noise |
| LASSO         | Feature selection through regularization |
| Random Forest | Identify non-linear patterns and feature importance |
| XGBoost       | Explore non-linear dependencies and interaction effects |

## Results Summary

| Model         | Training R² | Testing R² | Takeaways |
|---------------|-------------|------------|-----------|
| Linear        | 0.8786      | 0.8887     | Excellent interpretability |
| Ridge (α=0.1) | 0.8786      | 0.8887     | Strong performance, less overfitting risk |
| LASSO (α=0.0001) | 0.8786  | 0.8887     | Useful for sparse models |
| Random Forest | 0.8751      | 0.6346     | Overfitting likely |
| XGBoost       | 1.0000      | 0.7026     | Highly overfit on training set |

## Key Insights

- This project falls under **descriptive analytics**, focused on explaining the relationships between sales and influencing variables—not predicting future data or prescribing actions.
- **Price**, **ShelveLoc (Good)**, and **Age** emerged as the most influential factors across models.
- Regularized models provided slightly different coefficient magnitudes but aligned in direction with the base linear regression.
- Tree-based models helped validate the most important features via feature importance rankings.

## What Was Used

- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost)
- ISLP’s `Carseats` dataset
- Scikit-learn Pipelines, ColumnTransformer, GridSearchCV
