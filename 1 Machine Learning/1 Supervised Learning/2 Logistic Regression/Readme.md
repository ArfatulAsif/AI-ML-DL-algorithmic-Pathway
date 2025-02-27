# Linear Regression and Logistic Regression

## Linear Regression
Linear Regression is a **straight-line regression** technique that predicts the output in **continuous integer numbers** (e.g., 1, 23, 450, etc.).

### Linear Regression Function:
The equation for a simple linear regression model is:
```math
h = mX
```
Where:
- **h** = Hypothesis (Predicted Output)
- **m** = Coefficient of X
- **X** = Input feature

If we include a constant term, the function becomes:
```math
h = mX + c
```
Where:
- **c** = Constant (Intercept)

---

## Logistic Regression
Logistic Regression is an **advanced version of Linear Regression** used for classification problems.

It determines **categories or classes** (e.g., "Is an email spam?" → Yes/No). If the prediction is **1**, the email is classified as spam; otherwise, it is spamless.

### Characteristics of Logistic Regression:
- The predicted range is **between 0 and 1**.
- There is a **threshold** value of **0.5**:
  - If the predicted value **≥ 0.5**, it is classified as **1**.
  - If the predicted value **< 0.5**, it is classified as **0**.

### Logistic (Sigmoid) Function:
The sigmoid function is used to squash the output between **0 and 1**:
```math
h = \frac{1}{1 + e^{-z}}
```
Where:
- **z = mX** (Linear transformation of input)

This function ensures that the output is a probability value between **0 and 1**, making it suitable for binary classification tasks.

---

# Linear to Logistic Regression Conversion

## Linear Regression Equation
A standard linear regression model predicts an output \( y \) as:

$$
y = b_0 + b_1x_1 + b_2x_2 + \dots + b_nx_n
$$

where:
-  **y** is the predicted output, which can take any real value.
  
## Logistic Regression Transformation
Logistic regression is used for classification (where the output should be between 0 and 1). To achieve this, we apply the **sigmoid function**:

$$
p = \frac{1}{1 + e^{-(b_0 + b_1x_1 + b_2x_2 + \dots + b_nx_n)}}
$$

where:
- **p** is the probability that the outcome belongs to class **1** (e.g., success, positive classification).

## Expressing Probability in Terms of Odds
Rewriting **p** in terms of **odds**:

$$
\frac{p}{1 - p} = e^{(b_0 + b_1x_1 + b_2x_2 + \dots + b_nx_n)}
$$

Taking the natural logarithm on both sides:

$$
\log\left(\frac{p}{1 - p}\right) = b_0 + b_1x_1 + b_2x_2 + \dots + b_nx_n
$$

This equation represents the **log-odds (logit) function**, which is the foundation of logistic regression.

## Final Logistic Regression Equation

$$
p = \frac{1}{1 + e^{-(b_0 + b_1x_1 + b_2x_2 + \dots + b_nx_n)}}
$$

---

### Summary
| Regression Type   | Output Type      | Function         | Usage Example  |
|------------------|----------------|----------------|----------------|
| Linear Regression | Continuous values (1, 23, 450...) | `h = mX + c` | Predicting house prices |
| Logistic Regression | Binary classification (0 or 1) | `h = 1 / (1 + e^-z)` | Email spam detection |
