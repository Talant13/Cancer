# Cancer
Cancer analysis

# Waht are the best ways to deal with missing data
import pandas as pd

df = pd.read_csv("./cancer-risk-factors.csv")

# Cancer risk factors data set
df.info()
# Check if it has missing values
df.isnull().sum()
### Cancer_Type, Smoking, Risk_Level columns have a missing values
df.tail()
#### Cancer_Type -> Categorical
#### Smoking -> Numerical
#### Risk_Level -> Categorial
# Let`s consider all possible ways to deal with missing values
## 1. Deleting the columns with missing data
### Not a good approach we need any possible data
## 2. Deleting the row with missing data
### Not a good approach we need any possible data
## 3. Imputing missing values with mean/median
df['Smoking_mean'] = df['Smoking'].fillna(df['Smoking'].mean())

print(df.loc[df['Smoking'].isna(), ['Patient_ID', 'Smoking', "Smoking_mean"]])
![image.png](attachment:image.png)


As we can see mean data is far away from original file data. 

## 3.1 Imputing missing values with mean/median of group
In order to that we need to understand what group is highly correlated with "Smoking" column
import seaborn as sns
import matplotlib.pyplot as plt


df['Cancer_Type'] = df['Cancer_Type'].fillna('Unknown')

# One-Hot
cancer_ohe = pd.get_dummies(df['Cancer_Type'], prefix='Cancer', dtype='int')
df = pd.concat([df, cancer_ohe], axis=1)

plt.figure(figsize=(20, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

df['Smoking_mean_group'] = (
    df.groupby(['Air_Pollution', 'Risk_Level', "Cancer_Lung"])['Smoking']
      .transform('mean')
)

print(df.loc[df['Smoking'].isna(), ['Patient_ID', 'Smoking', "Smoking_mean", "Smoking_mean_group"]])
![image.png](attachment:image.png)
So here we can see that Group mean worked better than regular mean. 
##### Example:   Patient_ID LU0004, LU0005, LU0006 have smoking_group_mean -> 9, 9, 9
##### Real data: Patient_ID LU0004, LU0005, LU0006 have smoking score -> 10, 10,10
#### BUT
##### Here we can see that Patient_ID LU0001, LU0008 has smoking_group_mean -> 3.5, 5
##### Real data:           Patient_ID LU0001 has smoking value as 8, 9

#### More such cases might affect on future analysis!!!
## 4. Forward Fill and Backward Fill
We are not gonna even consider that
## 5. Interpolation
df['Smoking_interpolated'] = df['Smoking'].interpolate(method='linear', limit_direction='both')

print(df.loc[df['Smoking'].isna(), ['Patient_ID', 'Smoking', "Smoking_mean", "Smoking_mean_group", "Smoking_interpolated"]])
df.info()

from sklearn.impute import SimpleImputer

test_data = df[df["Smoking"].isnull()]
traindf   = df[df["Smoking"].notnull()]

# Целевая переменная
y_train = traindf["Smoking"]

# Список признаков (исключаем ненужные колонки)
feature_cols = [c for c in df.columns if c not in ["Smoking", "Patient_ID", "Cancer_Type", "Risk_Level"]]

# Формируем X_train и X_test
X_train = traindf[feature_cols].replace([np.inf, -np.inf], np.nan)
X_test  = test_data[feature_cols].replace([np.inf, -np.inf], np.nan)

# Импутация пропусков медианой
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed  = imputer.transform(X_test)

# Обучение модели
model = LinearRegression()
model.fit(X_train_imputed, y_train)

# Предсказание
y_pred = model.predict(X_test_imputed)

# Записываем предсказания обратно в исходный DataFrame

df["Smoking_pred"] = np.nan
df.loc[test_data.index, "Smoking_pred"] = y_pred

# (Необязательно) 9) Если хотите единый столбец без пропусков — оставьте исходные значения и подставьте предсказания вместо NaN
df["Smoking_filled"] = df["Smoking"].copy()
df.loc[test_data.index, "Smoking_filled"] = y_pred


df.head()
