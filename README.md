# 🔬 Cancer Risk Factors — Missing Data Handling with Python
 
A practical deep-dive into **7 strategies for handling missing values**, applied to a real-world cancer risk factors dataset. Each method is evaluated and compared to ground truth, making this a useful reference for data preprocessing decisions.
 
> 📖 Inspired by: [Handling Missing Values — 7 Methods](https://medium.com/@pingsubhak/handling-missing-values-in-dataset-7-methods-that-you-need-to-know-5067d4e32b62)  
> 📦 Dataset: [Cancer Risk Factors Dataset — Kaggle](https://www.kaggle.com/datasets/tarekmasryo/cancer-risk-factors-dataset/code)
 
---
 
## 📁 Dataset
 
**File:** `cancer-risk-factors.csv`
 
**Columns with missing values:**
 
| Column | Type | Missing Strategy Applied |
|--------|------|--------------------------|
| `Cancer_Type` | Categorical | Filled with `"Unknown"` |
| `Smoking` | Numerical | Multiple strategies compared |
| `Risk_Level` | Categorical | Analyzed |
 
---
 
## 🧪 Methods Compared
 
### ❌ 1 & 2 — Drop Columns / Drop Rows
Rejected. Any data loss is unacceptable in a medical dataset.
 
---
 
### ⚠️ 3 — Global Mean Imputation
```python
df['Smoking'].fillna(df['Smoking'].mean())
```
**Result:** Poor. The imputed mean was far from actual values for most patients.
 
---
 
### ✅ 3.1 — Group Mean Imputation
Identified the features most correlated with `Smoking` via a correlation heatmap (after one-hot encoding `Cancer_Type`).
 
**Best correlated group:** `Air_Pollution` + `Risk_Level` + `Cancer_Lung`
 
```python
df.groupby(['Air_Pollution', 'Risk_Level', 'Cancer_Lung'])['Smoking'].transform('mean')
```
 
**Result:** Much better than global mean — but edge cases exist.  
Example: `LU0001` had a group mean of `3.5` vs. actual `8` — a significant error.
 
---
 
### ❌ 4 — Forward / Backward Fill
Not considered. Patient records are not time-ordered; sequential filling has no logical basis here.
 
---
 
### 🔁 5 — Linear Interpolation
```python
df['Smoking'].interpolate(method='linear', limit_direction='both')
```
Applied as a baseline numerical approach.
 
---
 
### 🤖 6 — ML-Based Imputation (Linear Regression)
Used all non-missing rows to train a `LinearRegression` model, then predicted missing `Smoking` values.
 
```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
 
# Train on rows where Smoking is known
model = LinearRegression()
model.fit(X_train_imputed, y_train)
 
# Predict missing values
df.loc[missing_idx, 'Smoking_filled'] = model.predict(X_test_imputed)
```
 
**Feature preprocessing:** Median imputation via `SimpleImputer` for any remaining NaNs in features; `inf` values replaced before fitting.
 
---
 
## 📊 Method Comparison Summary
 
| Method | Accuracy | Risk | Recommended? |
|--------|----------|------|--------------|
| Drop rows/columns | — | High data loss | ❌ No |
| Global mean | Low | Distorts distribution | ⚠️ Last resort |
| Group mean | Medium | Edge cases possible | ✅ Good baseline |
| Forward/backward fill | N/A | Logically invalid here | ❌ No |
| Interpolation | Medium | Assumes linearity | ✅ For ordered data |
| ML (Linear Regression) | High | Needs sufficient training data | ✅ Recommended |
 
---
 
## 🛠️ Tech Stack
 
- **Python 3.x**
- `pandas` — data manipulation & imputation
- `numpy` — numerical operations
- `seaborn` / `matplotlib` — correlation heatmap & visualization
- `scikit-learn` — `SimpleImputer`, `LinearRegression`
---

## 💡 Key Takeaway
 
> There is no universal best method. The right strategy depends on the column type, data distribution, and how much accuracy matters downstream. For numerical medical data, **ML-based imputation** consistently outperforms simpler strategies — especially when correlations between features are strong.
 
---
