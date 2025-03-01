## Titanic Survival Prediction with SVM

This notebook demonstrates how to build a Support Vector Machine (SVM) model to predict survival on the Titanic dataset.

## 1. Importing Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

## 2. Importing the Data and Overview

```python
train_df = pd.read_csv('../input/titanic/train.csv')

train_df.head()

train_df.info()

train_df.shape

train_df.describe()

train_df['Survived'].value_counts()

# survived = 1
# didn't survive = 0
```

## 3. Exploratory Data Analysis (EDA)

### 3.1. Survival Count

```python
sns.countplot(data=train_df, x='Survived')
# survived = 1
# didn't survive = 0
```

### 3.2. Age vs. Survival Boxplot

```python
sns.boxplot(data=train_df, x='Survived', y='Age')
# survived = 1
# didn't survive = 0
```

### 3.3. Correlation Heatmap

```python
sns.heatmap(train_df.corr(), annot=True)
```

## 4. Data Preparation

### 4.1. Handling Null Values

```python
train_df.isnull().sum()

train_df["Age"].fillna(train_df["Age"].mean(), inplace=True)

train_df['Sex'] = train_df['Sex'].replace('male', 0)
train_df['Sex'] = train_df['Sex'].replace('female', 1)

train_df.drop(['Name', 'PassengerId', 'Fare', 'Ticket', 'Embarked', 'Cabin'], axis=1, inplace=True)

train_df.isnull().sum()

missing = train_df.isnull().sum().sort_values(ascending=False)
missing = missing.drop(missing[missing == 0].index)
missing
```

### 4.2. Defining Features and Labels

```python
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']
```

### 4.3. Splitting the Dataset

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```

### 4.4. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)
```

## 5. Training the Model (Basic SVM)

```python
from sklearn.svm import SVC

SVC_model = SVC()

SVC_model.fit(scaled_X_train, y_train)
```

## 6. Predicting and Evaluating the Basic Model

```python
y_pred = SVC_model.predict(scaled_X_test)

from sklearn.metrics import classification_report, confusion_matrix

confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))
```

## 7. Hyperparameter Tuning with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

svm = SVC()
grid_parameters = {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid_search = GridSearchCV(svm, grid_parameters, cv=5)

grid_search.fit(X_train, y_train)

grid_search.best_estimator_

grid_search.best_params_

y_pred_grid = grid_search.predict(X_test)

confusion_matrix(y_test, y_pred_grid)

print(classification_report(y_test, y_pred_grid))
