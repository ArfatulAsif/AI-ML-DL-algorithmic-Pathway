
# Support Vector Machines (SVM)

Support Vector Machines (SVM) are a powerful, supervised learning algorithm used primarily for classification, but also applicable to regression problems. SVMs are known for their ability to handle high-dimensional data and non-linear relationships effectively, thanks to the "kernel trick." They are a type of discriminative classifier, meaning they explicitly define a decision boundary.

## Hyperplanes and Margins

The core concept of SVM is finding the optimal *hyperplane* that best separates different classes of data. A hyperplane is a decision boundary: a line in 2D, a plane in 3D, and a higher-dimensional equivalent for more features. SVM doesn't just find *any* separating hyperplane; it aims for the one with the *maximum margin*. The margin is the distance between the hyperplane and the closest data points from each class. These closest points are called *support vectors*. A larger margin generally indicates a more robust and generalizable classifier.
![](https://github.com/ArfatulAsif/AI-ML-DL-algorithmic-Pathway/blob/main/1%20Machine%20Learning/1%20Supervised%20Learning/5%20SVM/mapping%20to%20higher%20dimension_0.png)
![](https://github.com/ArfatulAsif/AI-ML-DL-algorithmic-Pathway/blob/main/1%20Machine%20Learning/1%20Supervised%20Learning/5%20SVM/mapping%20to%20higher%20dimension%201.png)
## Now lets explore higher dimensions!!
![](https://github.com/ArfatulAsif/AI-ML-DL-algorithmic-Pathway/blob/main/1%20Machine%20Learning/1%20Supervised%20Learning/5%20SVM/mapping%20to%20higher%20dimension%202.png)
![](https://github.com/ArfatulAsif/AI-ML-DL-algorithmic-Pathway/blob/main/1%20Machine%20Learning/1%20Supervised%20Learning/5%20SVM/Projecting%20back%20to%202D%20space.png)
## Support Vectors

Support vectors are the critical data points that define the margin and, consequently, the optimal hyperplane. They are the points "closest" to the decision boundary. Only these support vectors are needed to define the classifier; other data points further away are less relevant. This makes SVM relatively memory-efficient.

## Non-Linear Separability and the Kernel Trick

Many real-world datasets aren't linearly separable (i.e., you can't draw a straight line/plane to separate the classes). This is where the "kernel trick" comes in. A *kernel* is a function that implicitly transforms the data into a higher-dimensional space where linear separation *is* possible. The SVM algorithm doesn't explicitly perform this transformation; the kernel function calculates the dot products (relationships) between data points *as if* they were in the higher-dimensional space. This avoids the computational cost of actually transforming the data.

## Common Kernel Functions

*   **Linear Kernel:** `K(x, xi) = sum(x * xi)`. Used for linearly separable data. Essentially, it's just a dot product.
*   **Polynomial Kernel:** `K(x,xi) = (1 + sum(x * xi))^d`. Introduces non-linearity; `d` is the degree of the polynomial.
*   **Radial Basis Function (RBF) Kernel:** `K(x,xi) = exp(-gamma * sum((x – xi)^2))`. A very popular and powerful kernel. `gamma` controls the influence of each data point; a smaller `gamma` gives each point a wider influence. A good default value for gamma is 0.1.

## Classification Problems

SVMs are primarily used for classification. Given a set of labeled training data (e.g., images of cats and dogs, with each image labeled as "cat" or "dog"), the SVM finds the optimal hyperplane to separate the classes. When a new, unlabeled data point (a new image) is presented, the SVM determines which side of the hyperplane it falls on and assigns the corresponding class label.

## Regression Problems

Although less common, SVMs can also be used for regression. Instead of finding a separating hyperplane, the goal is to find a hyperplane that best fits the data, minimizing the error within a certain margin (similar to a "tube" around the hyperplane).

## Training Phase

SVM has a distinct training phase where it finds the optimal hyperplane and identifies the support vectors. This training phase can be computationally intensive, especially for large datasets. However, once the model is trained, making predictions is relatively fast.

## An Example:

Imagine a dataset of points in 2D space, representing two different types of flowers (e.g., Iris species) based on sepal length and width. Some points are labeled "Setosa," and others are labeled "Versicolor."

*   **Linearly Separable Case:** If a straight line can clearly separate the Setosa points from the Versicolor points, a linear SVM can be used. The algorithm finds the line that maximizes the margin between the closest points of each class.

*   **Non-Linearly Separable Case:** If the points are arranged in a way that a straight line *cannot* separate them (e.g., a circular arrangement), a kernel trick is needed. An RBF kernel might be used to implicitly map the data to a higher dimension where a hyperplane *can* separate the classes.

Once the SVM is trained (either with a linear or non-linear kernel), a new flower with its sepal length and width can be classified by determining which side of the decision boundary it falls on.
## Optimization Problem: Detailed Explanation

The core of the Support Vector Machine (SVM) algorithm lies in solving a constrained optimization problem. Let's break it down step-by-step:

**1. The Goal: Maximize the Margin (Intuitively)**

Imagine separating two groups of points (positive and negative classes) with a line (in 2D) or a hyperplane (in higher dimensions).  Many lines/hyperplanes *could* separate the data.  SVM aims to find the "best" separator – the one that leaves the *most* space (the *margin*) between itself and the closest points from each class. A larger margin implies a more robust and generalizable classifier.

**2. Representing the Hyperplane and Distance**

*   **Hyperplane Equation:**  A hyperplane is defined by:

    ```
    w · x + b = 0
    ```

    where:
    *   `w`: A vector of weights (coefficients), *normal* (perpendicular) to the hyperplane.  Its direction determines the hyperplane's orientation.
    *   `x`: A data point (a vector of features).
    *   `b`: The bias term (a scalar), shifting the hyperplane away from the origin.

*   **Signed Distance:** The *signed distance* of a point `x_i` from the hyperplane is:

    ```
    distance = (w · x_i + b) / ||w||
    ```

    *   The sign indicates which side of the hyperplane the point is on (positive or negative).
    *   `||w||` is the Euclidean norm (magnitude) of `w`, normalizing the distance.

**3. Why Minimize ||w||? (The Connection to the Margin)**

We want to *maximize* the margin, but the optimization problem minimizes `||w||`. Here's why:

*   **The Margin is 1/||w||:** This is the key. The distance from the hyperplane to the *closest* point (a support vector) is `1/||w||` (provided the constraints below are met).  Therefore, maximizing the margin is equivalent to minimizing `||w||`.
    We can represent the distance to the nearest positive example as:
        `(w · x+ + b) / ||w|| = 1 / ||w||`
    And for the negative examples as:
    `(w · x- + b) / ||w|| = -1 / ||w||`

*   **Why (1/2)||w||² instead of ||w||?** Minimizing `||w||` and minimizing `(1/2)||w||²` give the same optimal `w`.  The squared term, `||w||²`, is mathematically more convenient (differentiable, leading to a quadratic programming problem with efficient solutions). The `1/2` is a constant for simplification.

**4. The Constraints: Ensuring Correct Classification and the 1/||w|| Margin**

The constraints are crucial. They:

*   **Ensure Correct Classification:** All data points must be on the *correct* side of the decision boundary.
*   **Define the Margin:** They force the distance from the hyperplane to the *closest* points (support vectors) to be *exactly* `1/||w||`.

Combined Constraint Form (using class labels `y_i`):
y_i (w · x_i + b) ≥ 1    for all data points i
Where:

*   `y_i = +1` for points in the positive class.
*   `y_i = -1` for points in the negative class.

Breakdown of the Constraint:

*   **`w · x_i + b`:** The (unnormalized) signed distance of `x_i` from the hyperplane.

*   **`y_i (w · x_i + b)`:** Multiplying by `y_i` ensures:
    *   **Positive Class (y_i = +1):** If `x_i` is correctly classified, `w · x_i + b` must be positive.  The product is positive.
    *   **Negative Class (y_i = -1):** If `x_i` is correctly classified, `w · x_i + b` must be negative. The product is *also* positive (negative times negative).

*   **`≥ 1`:**  This is critical. It's not just `≥ 0`.  The `≥ 1` ensures:
    *   **Correct Classification:** The product `y_i (w · x_i + b)` being ≥ 1 guarantees it's positive, so the point is on the correct side.
    *   **Defines the Margin:** This `1` is directly tied to the `1/||w||` margin. It forces the *closest* points (support vectors) to have a (scaled) signed distance of at least 1.  Support vectors will have `y_i (w · x_i + b) = 1`.

**In Summary: The Optimization Problem**

*   **Objective:** Minimize `(1/2)||w||²` (equivalent to maximizing the margin, `1/||w||`).
*   **Constraints:** `y_i (w · x_i + b) ≥ 1` for all data points `i`.  These ensure correct classification and define the margin.

This is a *constrained optimization problem*. We minimize `(1/2)||w||²` subject to the constraints.  The solution yields the `w` and `b` defining the optimal separating hyperplane with maximum margin. Support vectors are the data points where the constraint holds with equality.

These constraints ensure that all data points fall into the correct class and that the margin from the nearest point is exactly 1/||w||.
## Hyperparameter Tuning

*   **C (Regularization):** Controls the trade-off between maximizing the margin and minimizing classification errors on the *training* data. A smaller C allows for a wider margin but may misclassify more training points. A larger C forces a smaller margin but tries to classify training points more accurately.
*   **Kernel:** Choosing the appropriate kernel (linear, polynomial, RBF, etc.) is crucial and depends on the data.
*   **Gamma (for RBF kernel):** Controls the influence of each data point. Smaller gamma = wider influence; larger gamma = tighter influence (can lead to overfitting).
*    **Degree(d) (for polynomial kernel):** specify the d.

## Advantages of SVM

*   Effective in high-dimensional spaces.
*   Relatively memory efficient (uses only support vectors).
*   Versatile: different kernel functions can be used.
*   Good accuracy in many applications.

## Disadvantages of SVM

*   Can be computationally expensive to train, especially with large datasets.
*   Prone to overfitting if hyperparameters (C, gamma, kernel) aren't chosen carefully.
*   Performance can degrade with overlapping classes.
*   Doesn't directly provide probability estimates (unlike, say, logistic regression).

## Implementation Overview of SVM using Python (scikit-learn)

1.  **Import Libraries:** Import necessary libraries, including `SVC` (for classification) or `SVR` (for regression) from `sklearn.svm`. Also, import libraries like `pandas`, `numpy`, and `matplotlib`.

2.  **Load Data:** Load the dataset into a DataFrame (e.g., using `pandas`).

3.  **Preprocessing:**
    *   Handle missing values if necessary.
    *   Encode categorical features (if any) into numerical representations (e.g., using one-hot encoding).
    *   Scale features (important for SVM!). Standardization (using `StandardScaler`) is often recommended.

4.  **Define the Model:** Create an instance of the SVM model (`SVC` or `SVR`), specifying the kernel and other hyperparameters:
    ```python
    from sklearn import svm
    model = svm.SVC(kernel='linear', C=1.0, gamma=0.1)  # Example
    ```

5.  **Model Training:** Fit the model to the training data using the `.fit()` method:
    ```python
    model.fit(X_train, y_train)
    ```

6.  **Cross-Validation (Optional but Recommended):** Use cross-validation (e.g., `cross_val_score`) to get a more reliable estimate of the model's performance.

7.  **Prediction:** Use the `.predict()` method to make predictions on new data:
    ```python
    y_pred = model.predict(X_test)
    ```
8. **Evaluation:** Evaluate Model using, precision, recall and accuracy.
```python
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?print("Recall:",metrics.recall_score(y_test, y_pred))
