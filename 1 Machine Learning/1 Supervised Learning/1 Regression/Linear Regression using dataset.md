Here the following things will cover: <br>

1. **Understanding the dataset**  
2. **Fitting a Linear Regression model**  
3. **Making predictions**  
4. **Evaluating the model**  
5. **Visualizing the results**  
6. **Handling multiple features (Multiple Linear Regression)**  
7. **Checking model assumptions**  
8. **Polynomial Regression (for nonlinear relationships)**  

---

## **1. Understanding the Dataset**  

We'll use a simple dataset where:  
- **Independent Variable (X):** Hours studied  
- **Dependent Variable (y):** Exam scores  

| Hours Studied (X) | Exam Score (y) |
|-------------------|---------------|
| 1                | 50            |
| 2                | 60            |
| 3                | 70            |
| 4                | 80            |

This dataset shows a clear **linear relationship**: as study hours increase, exam scores also increase.

---

## **2. Fitting a Linear Regression Model**  

Let's fit a simple **Linear Regression** model using `scikit-learn`:  

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Example dataset
X = np.array([[1], [2], [3], [4]])  # Hours studied
y = np.array([50, 60, 70, 80])      # Exam scores

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Get model parameters
print("Intercept (β0):", model.intercept_)
print("Slope (β1):", model.coef_[0])
```

### **Explanation**  
- `model.fit(X, y)`: Trains the linear regression model.  
- `model.intercept_`: The **y-intercept** (β₀), which is the predicted score when X = 0.  
- `model.coef_`: The **slope** (β₁), which tells us how much `y` changes for every unit increase in `X`.  

For our dataset, the output will be:  
```
Intercept (β0): 40.0  
Slope (β1): 10.0  
```
So, the equation of our regression line is:  
\[
y = 40 + 10x
\]
This means:  
- A student who studies **0 hours** is expected to score **40 marks**.  
- For each additional hour of study, the score increases by **10 points**.

---

## **3. Making Predictions**  

Now, let's predict the exam score for a student who studies **5 hours**:

```python
predicted_score = model.predict([[5]])
print("Predicted score for 5 hours of study:", predicted_score[0])
```

**Output:**  
```
Predicted score for 5 hours of study: 90.0
```
This matches our equation:
\[
y = 40 + 10(5) = 90
\]

---

## **4. Evaluating the Model**  

We can evaluate how well our model fits the data using **R-squared** (\( R^2 \)):

```python
r2_score = model.score(X, y)
print("R-squared:", r2_score)
```

Since our dataset follows a perfect linear pattern, \( R^2 = 1.0 \), meaning **100% of the variance** in exam scores is explained by study hours.

---

## **5. Visualizing the Results**  

Let's plot the dataset and regression line:

```python
import matplotlib.pyplot as plt

# Scatter plot of actual data
plt.scatter(X, y, color='blue', label="Actual Data")

# Regression line
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.show()
```

This graph visually confirms the linear relationship between study hours and exam scores.

---

## **6. Handling Multiple Features (Multiple Linear Regression)**  

Let's extend our model to **Multiple Linear Regression (MLR)**. We'll now include:  
- **Hours studied** (X1)  
- **Number of practice tests taken** (X2)  

### **Example Dataset**  

| Hours Studied (X1) | Practice Tests (X2) | Exam Score (y) |
|-------------------|--------------------|---------------|
| 1                | 2                  | 52            |
| 2                | 3                  | 58            |
| 3                | 5                  | 70            |
| 4                | 7                  | 80            |

### **Fitting Multiple Linear Regression**  

```python
X_multi = np.array([[1, 2], [2, 3], [3, 5], [4, 7]])  # Features: Hours studied & Practice tests
y_multi = np.array([52, 58, 70, 80])  # Exam scores

# Train the model
model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

# Display coefficients
print("Intercept:", model_multi.intercept_)
print("Coefficients:", model_multi.coef_)

# Predict exam score for a student who studied 5 hours and took 8 practice tests
predicted_multi = model_multi.predict([[5, 8]])
print("Predicted Score:", predicted_multi[0])
```

---

## **7. Checking Model Assumptions**  

Linear regression makes several assumptions:  
- **Linearity:** The relationship between X and y is linear (scatter plots help check this).  
- **Independence:** No dependency between observations.  
- **Homoscedasticity:** Residuals (errors) have constant variance.  
- **Normality of Residuals:** Residuals should be normally distributed.  

To check residuals:  

```python
import seaborn as sns

residuals = y - model.predict(X)
sns.histplot(residuals, kde=True)
plt.xlabel("Residuals")
plt.title("Residual Distribution")
plt.show()
```

---

## **8. Polynomial Regression (For Non-Linear Data)**  

If the relationship is **curved**, **Polynomial Regression** is needed:

```python
from sklearn.preprocessing import PolynomialFeatures

# Transform X into polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train polynomial regression model
model_poly = LinearRegression()
model_poly.fit(X_poly, y)

# Predict and plot results
plt.scatter(X, y, color='blue')
plt.plot(X, model_poly.predict(X_poly), color='red', linewidth=2)
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Polynomial Regression")
plt.show()
```

---



### **Key Takeaways**  
- **Linear Regression** is simple, interpretable, and widely used in statistics and machine learning.  
- **MLR** allows multiple factors to influence the prediction.  
- **Checking assumptions** ensures the model is valid.  
- **Polynomial Regression** handles curved relationships.  

