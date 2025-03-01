

---

### What is Linear Regression?

Linear Regression is one of the simplest and most widely used machine learning algorithms. It is used to **predict a numerical value** (like house prices, exam scores, or temperature) based on one or more input features (like size of the house, hours studied, or humidity).

Think of it as drawing a straight line through data points to find the best relationship between inputs (features) and outputs (target values).

---

### Key Idea: The Line of Best Fit

Imagine you have some data points on a graph:

- On the X-axis (horizontal), you have an input feature (e.g., "hours studied").
- On the Y-axis (vertical), you have the output (e.g., "exam score").

If you plot these points, they might look scattered. Linear Regression tries to find the **best straight line** that fits these points. This line is called the **line of best fit**.

The equation of this line looks like this:

$$
Y = mX + c
$$

Where:
- $Y$ = Predicted output (e.g., exam score)
- $X$ = Input feature (e.g., hours studied)
- $m$ = Slope of the line (how steep the line is)
- $c$ = Intercept (where the line crosses the Y-axis)

  <img src="fig1.png"  width="450">

<br>

  **[Derivation](#derivation)**

---

### How Does It Work?

1. **Goal**: Find the best values for $m$ (slope) and $c$ (intercept) so that the line fits the data well.
2. **Error Calculation**: The algorithm calculates how far each data point is from the line. This distance is called the **error**.
3. **Minimize Error**: The algorithm adjusts $m$ and $c$ to minimize the total error. This process is called **optimization**.

The most common method to minimize the error is called **Least Squares**, which minimizes the sum of the squared distances between the actual data points and the predicted values on the line.

---

### Example: Predicting Exam Scores

Let’s say you want to predict a student's exam score based on the number of hours they studied.

| Hours Studied ($X$) | Exam Score ($Y$) |
|----------------------|-------------------|
| 1                    | 50                |
| 2                    | 60                |
| 3                    | 70                |
| 4                    | 80                |

1. Plot these points on a graph.
2. Draw a straight line that best fits these points.
3. Use the equation of the line ($Y = mX + c$) to predict scores for new values of $X$ (hours studied).

For example, if the line equation is $Y = 10X + 40$:
- If a student studies for 5 hours ($X = 5$), their predicted score would be:
  $$
  Y = 10(5) + 40 = 90
  $$

---

### Why Is It Called "Linear"?

It’s called **linear** because the relationship between the input ($X$) and output ($Y$) is represented by a straight line. If you have multiple input features (e.g., hours studied, sleep hours, etc.), the concept extends to higher dimensions, but the idea remains the same: finding the best-fitting "plane" or "hyperplane."

---

### When to Use Linear Regression?

- When the relationship between the input(s) and output is roughly linear.
- When you need to predict a continuous numerical value (not categories like "yes/no").
- When the dataset is not too complex (for complex data, other algorithms may work better).

---

### Summary

Linear Regression is like finding the best straight line that describes how one thing (input) affects another (output). It’s simple, easy to understand, and a great starting point for learning machine learning.

---

### experimenting with a small dataset using Python libraries like `scikit-learn`.

 a quick example:

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Example data
X = np.array([[1], [2], [3], [4]])  # Hours studied
y = np.array([50, 60, 70, 80])      # Exam scores

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict for 5 hours of study
print(model.predict([[5]]))  # Output: [90.]
```



---


### <a id="derivation"></a>Derivation

#### Linear regression is a fundamental statistical method used to model the relationship between a dependent variable y and one or more independent variables x. In simple linear regression, we aim to find the best-fitting straight line that minimizes the error between the observed data points and the predicted values. The equation of the line is given by:

$$
y = \beta_0 + \beta_1 x
$$

where:
- y: Dependent variable (response)
- y: Independent variable (predictor)
- β0: Intercept (value of y when x = 0)
- β1: Slope (rate of change of y with respect to x)

The goal is to derive the values of β0 and β1 that minimize the sum of squared errors (SSE). Let’s go through the derivation step by step.

---

### The Sum of Squared Errors (SSE)
The error for a single data point is the difference between the observed value yi and the predicted value ȳi = β0 + β1xi. The total error is minimized by minimizing the sum of squared errors:


#### The Sum of Squared Errors (SSE) is defined as:

$$
SSE = \sum_{i=1}^n \left( y_i - (\beta_0 + \beta_1 x_i) \right)^2
$$

---

### Minimize SSE with Respect to β0 and β1
To find the values of β0 and β1 that minimize SSE, we take partial derivatives of SSE with respect to β0 and β1, set them to zero, and solve for β0 and β1.

#### Partial Derivative with Respect to β0 :
$$
\frac{\partial SSE}{\partial \beta_0} = \frac{\partial}{\partial \beta_0} \sum_{i=1}^n \left( y_i - (\beta_0 + \beta_1 x_i) \right)^2
$$

#### Using the chain rule:
$$
\frac{\partial SSE}{\partial \beta_0} = \sum_{i=1}^n 2 \left( y_i - (\beta_0 + \beta_1 x_i) \right)(-1)
$$

#### Simplifying:
$$
\frac{\partial SSE}{\partial \beta_0} = -2 \sum_{i=1}^n \left( y_i - \beta_0 - \beta_1 x_i \right)
$$

#### Setting ∂SSE/∂β0 = 0:
$$
\sum_{i=1}^n \left( y_i - \beta_0 - \beta_1 x_i \right) = 0
$$

#### Rearranging:
$$
\sum_{i=1}^n y_i = n \beta_0 + \beta_1 \sum_{i=1}^n x_i
$$

#### Divide through by n:
$$
\bar{y} = \beta_0 + \beta_1 \bar{x}
$$

#### This gives us the first equation:
$$
\beta_0 = \bar{y} - \beta_1 \bar{x}
$$

---

#### Partial Derivative with Respect to  β1 :
$$
\frac{\partial SSE}{\partial \beta_1} = \frac{\partial}{\partial \beta_1} \sum_{i=1}^n \left( y_i - (\beta_0 + \beta_1 x_i) \right)^2
$$

#### Using the chain rule: 
$$
\frac{\partial SSE}{\partial \beta_1} = \sum_{i=1}^n 2 \left( y_i - (\beta_0 + \beta_1 x_i) \right)(-x_i)
$$

#### Simplifying:
$$
\frac{\partial SSE}{\partial \beta_1} = -2 \sum_{i=1}^n x_i \left( y_i - \beta_0 - \beta_1 x_i \right)
$$

#### Set ∂SSE/∂β1 = 0 :
$$
\sum_{i=1}^n x_i \left( y_i - \beta_0 - \beta_1 x_i \right) = 0
$$

#### Substitute β0 = ȳ - β1x into the equation:
$$
\sum_{i=1}^n x_i \left( y_i - (\bar{y} - \beta_1 \bar{x}) - \beta_1 x_i \right) = 0
$$

#### Simplifying:
$$
\sum_{i=1}^n x_i \left( y_i - \bar{y} + \beta_1 \bar{x} - \beta_1 x_i \right) = 0
$$

#### Distribute x1:
$$
\sum_{i=1}^n x_i (y_i - \bar{y}) + \beta_1 \sum_{i=1}^n x_i (\bar{x} - x_i) = 0
$$

#### Split terms:
$$
\sum_{i=1}^n x_i (y_i - \bar{y}) = \beta_1 \sum_{i=1}^n x_i (x_i - \bar{x})
$$

#### Solve for β1:
$$
\beta_1 = \frac{\sum_{i=1}^n x_i (y_i - \bar{y})}{\sum_{i=1}^n x_i (x_i - \bar{x})}
$$

---

### Simplifying the Expressions
We can rewrite the numerator and denominator in terms of covariance and variance:
- #### Covariance:
$$ \text{Cov}(x, y) = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}) $$
- #### Variance:
$$ \text{Var}(x) = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2 $$

#### Thus:
$$
\beta_1 = \frac{\text{Cov}(x, y)}{\text{Var}(x)}
$$

#### And from earlier:
$$
\beta_0 = \bar{y} - \beta_1 \bar{x}
$$

---

### Final Result
#### The linear regression coefficients are:
$$
\boxed{\beta_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}}
$$
$$
\boxed{\beta_0 = \bar{y} - \beta_1 \bar{x}}
$$

These are the formulas for the slope β1 and intercept β0 of the best-fit line in simple linear regression.

- The working formula for linear regression

<img src="fig2.png"  width="350">

