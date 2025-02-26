# Linear Data Classification using Logistic Regression

## Introduction
This project implements an linear data classification model using Logistic Regression on the Social_Network_Ads dataset.

## Dataset Loading

```python
import pandas as pd
# Load CSV file
df = pd.read_csv("Social_Network_Ads.csv")  
df
```
## Dataset Overview

This dataset contains information about users, including their ID, gender, age, estimated salary, and whether they made a purchase. 

### Sample Data:

| User ID  | Gender | Age | Estimated Salary | Purchased |
|----------|--------|-----|------------------|-----------|
| 15624510 | Male   | 19  | 19000            | 0         |
| 15810944 | Male   | 35  | 20000            | 0         |
| 15668575 | Female | 26  | 43000            | 0         |
| 15603246 | Female | 27  | 57000            | 0         |
| 15804002 | Male   | 19  | 76000            | 0         |
| ...      | ...    | ... | ...              | ...       |
| 15691863 | Female | 46  | 41000            | 1         |
| 15706071 | Male   | 51  | 23000            | 1         |
| 15654296 | Female | 50  | 20000            | 1         |
| 15755018 | Male   | 36  | 33000            | 0         |
| 15594041 | Female | 49  | 36000            | 1         |

- **User ID:** Unique identifier for each user.
- **Gender:** Male or Female.
- **Age:** Age of the user.
- **Estimated Salary:** The estimated annual salary of the user.
- **Purchased:** Whether the user made a purchase (1 for Yes, 0 for No).

This dataset contains **400 rows** and **5 columns**.

## drop() used to delete a column
```python
df = df.drop(columns = 'User ID')
df
```
## Dataset Overview

This dataset contains information about users, including their gender, age, estimated salary, and whether they made a purchase.  

### Sample Data without 'User ID' column:

| Gender | Age | Estimated Salary | Purchased |
|--------|-----|------------------|-----------|
| Male   | 19  | 19000            | 0         |
| Male   | 35  | 20000            | 0         |
| Female | 26  | 43000            | 0         |
| Female | 27  | 57000            | 0         |
| Male   | 19  | 76000            | 0         |
| ...    | ... | ...              | ...       |
| Female | 46  | 41000            | 1         |
| Male   | 51  | 23000            | 1         |
| Female | 50  | 20000            | 1         |
| Male   | 36  | 33000            | 0         |
| Female | 49  | 36000            | 1         |

### Column Descriptions:
- **Gender:** Male or Female.
- **Age:** Age of the user.
- **Estimated Salary:** The estimated annual salary of the user.
- **Purchased:** Whether the user made a purchase (1 for Yes, 0 for No).

This dataset contains **400 rows** and **4 columns**.

 ## For check how many null value available.
```python
df.isnull().sum()   
```
## If want to drop null value command -> df.dropna()

## Sample Data

```python
Gender             0
Age                0
EstimatedSalary    0
Purchased          0
dtype: int64
```

## Now we should change all String to number as category style

```python
df['Gender'] = df['Gender'].astype('category')
df['Gender'] = df['Gender'].cat.codes
df
```

## Sample Data with 'Gender' column value change string to number

| Gender | Age | Estimated Salary | Purchased |
|--------|-----|------------------|-----------|
| 1 (Male)   | 19  | 19000            | 0         |
| 1 (Male)   | 35  | 20000            | 0         |
| 0 (Female) | 26  | 43000            | 0         |
| 0 (Female) | 27  | 57000            | 0         |
| 1 (Male)   | 19  | 76000            | 0         |
| ...    | ... | ...              | ...       |
| 0 (Female) | 46  | 41000            | 1         |
| 1 (Male)   | 51  | 23000            | 1         |
| 0 (Female) | 50  | 20000            | 1         |
| 1 (Male)   | 36  | 33000            | 0         |
| 0 (Female) | 49  | 36000            | 1         |

### Column Descriptions:
- **Gender:** Encoded as `1` for Male and `0` for Female.


## Choosing 'X'(Independent variable/variables)

```python
X = df.drop(columns = 'Purchased') 
X
```

## Sample Data of 'X'

| Gender | Age | Estimated Salary |
|--------|-----|------------------|
| 1 (Male)   | 19  | 19000            |
| 1 (Male)   | 35  | 20000            |
| 0 (Female) | 26  | 43000            |
| 0 (Female) | 27  | 57000            |
| 1 (Male)   | 19  | 76000            |
| ...    | ... | ...              |
| 0 (Female) | 46  | 41000            |
| 1 (Male)   | 51  | 23000            |
| 0 (Female) | 50  | 20000            |
| 1 (Male)   | 36  | 33000            |
| 0 (Female) | 49  | 36000            |

## Choosing 'Y'(Predicted Variable)

```python
Y = df['Purchased']
Y
```

## Sample Data of 'Y'

```python
0      0
1      0
2      0
3      0
4      0
      ..
395    1
396    1
397    1
398    0
399    1
Name: Purchased, Length: 400, dtype: int64
```

# Splitting the Training and Testing sets using train_test_split()

```python
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3, random_state = 21)
```

### test_size = 0.3 means 30% of dataset are using for test and 70% are using for train
### Random_state can be any integer, by using random_state we can find same train-test_split

```python
X_train
```

## Sample Data with only X_Training set

| Gender | Age | Estimated Salary |
|--------|-----|------------------|
| 1 (Male)   | 37  | 55000            |
| 1 (Male)   | 49  | 28000            |
| 1 (Male)   | 24  | 23000            |
| 0 (Female) | 35  | 44000            |
| 1 (Male)   | 31  | 18000            |
| ...    | ... | ...              |
| 1 (Male)   | 38  | 71000            |
| 1 (Male)   | 30  | 135000           |
| 0 (Female) | 35  | 77000            |
| 0 (Female) | 38  | 50000            |
| 0 (Female) | 52  | 114000           |

Contains **280 rows** and **3 columns**.

```python
X_test
```

## Sample Data with only X_Testing set

| Gender | Age | Estimated Salary |
|--------|-----|------------------|
| 0 (Female) | 26  | 35000  |
| 0 (Female) | 35  | 65000  |
| 1 (Male)   | 25  | 87000  |
| 0 (Female) | 35  | 60000  |
| 1 (Male)   | 30  | 49000  |
| ...    | ... | ...      |
| 0 (Female) | 45  | 22000  |
| 1 (Male)   | 29  | 75000  |
| 1 (Male)   | 38  | 61000  |
| 0 (Female) | 52  | 90000  |
| 1 (Male)   | 26  | 16000  |

Contains **120 rows** and **3 columns**.

```python
Y_train
```

## Sample Data with only Y_Training set

```python
113    0
26     1
178    0
95     0
29     0
      ..
368    0
48     1
260    0
312    0
207    0
Name: Purchased, Length: 280, dtype: int64
```

```python
Y_test
```

## Sample Data with only Y_Testing set

```python
106    0
9      0
61     0
224    0
37     0
      ..
23     1
157    0
349    0
255    1
180    0
Name: Purchased, Length: 120, dtype: int64
```

# For all numbers should convert into a fixed range.
## For converting do some Scalling for pre-processing


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Use fit_transform() to know the train sets 'standard deviation' and 'mean'
X_test_scaled = scaler.transform(X_test)
# Use only transform() so that model should not know anything about 'test set'
X_train_scaled
```

## Sample Data with only X_train_scaled set

```python
array([[ 0.97182532, -0.11728762, -0.48154649],
       [ 0.97182532,  1.06615502, -1.26799962],
       [ 0.97182532, -1.39935048, -1.41363908],
       [-1.02899151, -0.31452806, -0.80195332],
       [ 0.97182532, -0.70900894, -1.55927855],
       [ 0.97182532, -1.0048696 ,  0.21752295],
       [-1.02899151,  0.4744337 ,  1.79042919],
       [ 0.97182532, -0.21590784, -0.33590703],
       [-1.02899151, -1.0048696 ,  0.36316241],
       [-1.02899151, -0.31452806, -1.41363908],
       ...
```

```python
X_test_scaled
```

## Sample Data with only X_train_scaled set

```python
array([[-1.02899151, -1.20211004, -1.06410436],
       [-1.02899151, -0.31452806, -0.19026756],
       [ 0.97182532, -1.30073026,  0.45054609],
       [-1.02899151, -0.31452806, -0.33590703],
       [ 0.97182532, -0.80762916, -0.65631385],
       [-1.02899151,  1.16477524,  0.47967399],
       [-1.02899151,  1.06615502,  2.02345234],
       [-1.02899151, -0.90624938,  0.33403452],
       [-1.02899151,  1.55925613,  1.06223185],
       [-1.02899151,  1.46063591,  2.08170812],
       [-1.02899151,  0.07995282,  0.21752295],
       [-1.02899151, -1.0048696 , -1.00584857],
       [-1.02899151,  0.77029436, -1.44276698],
       [ 0.97182532,  0.07995282,  0.71269713],
       [ 0.97182532,  0.27719326,  0.01362769],
       [-1.02899151, -1.79383136,  0.30490663],
       ...
```

## Model Training ->
```python
from sklearn.linear_model import LogisticRegression
logistic_reg = LogisticRegression(random_state = 0).fit(X_train_scaled,Y_train) 
logistic_reg.predict(X_train_scaled) 
```

## Sample Data with predicted X_train_scaled set

```python 
array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
       1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0,
       0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1,
       0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0,
       0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
       1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1,
       0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0,
       1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
       0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=int64)
```

```python
logistic_reg.predict(X_test_scaled)
``` 

## Sample Data with predicted X_test_scaled set

```python
array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
       0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
       0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=int64)
```

## Accuracy Testing ->

```python
logistic_reg.score(X_train_scaled,Y_train) 
```

```python
output = 0.8214285714285714 means 82.1428..% accurate
```

```python
logistic_reg.score(X_test_scaled,Y_test) 
```

```python
output = 0.85 means 85% accurate
```

**Note:** The model is said to be best the more close accuracy match with **train** and **test** 

## We can add some more features for increasing accuracy =>

C = regularization parameter, Regularization is a technique used in machine learning to prevent overfitting. 
- Overfitting happens when a model learns the training data too well, including the noise and outliers, which causes it to perform poorly on new data. In simple terms, regularization adds a penalty to the model for being too complex, encouraging it to stay simpler and more general. This way, it’s less likely to make extreme predictions based on the noise in the data.
- '1' its default value.
fit_intercept = is the intersaction (y = mx + c). True means there should be intersaction.

```python
# Model Trained 
logistic_reg1 = LogisticRegression(random_state = 0, C = 1, fit_intercept = True).fit(X_train_scaled,Y_train)
# Calculating Accuracy
logistic_reg1.score(X_train_scaled,Y_train)  
```

```python
output = 0.8214285714285714 or 82.14....%
```

```python
# Calculating Accuracy
logistic_reg1.score(X_test_scaled,Y_test)  
```

```python
output = 0.85 or 85%
```

```python
# Model Trained 
logistic_reg2 = LogisticRegression(random_state = 0, C = 50, fit_intercept = True).fit(X_train_scaled,Y_train)
# Calculating Accuracy
logistic_reg2.score(X_train_scaled,Y_train)  
```

```python
output = 0.8321428571428572 or 83.21....%
```

```python
# Calculating Accuracy
logistic_reg2.score(X_test_scaled,Y_test)  
```

```python
output = 0.8416666666666667 or 84.1666...%
```
