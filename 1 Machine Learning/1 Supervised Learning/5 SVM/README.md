
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

Okay, here's a simplified version of the SVM optimization problem, focusing on the core concepts without getting bogged down in every detail:

## SVM Optimization: The Simplified Version

**The Big Picture:**

SVM tries to find the best line (or hyperplane in higher dimensions) to separate two classes of data.  "Best" means the line that has the largest possible *margin* – the biggest gap between the line and the closest points from each class.

**What We Want:**

*   **Maximum Margin:**  We want the fattest possible "street" between the two classes, with the decision boundary running right down the middle of the street.

**How We Achieve It (Mathematically):**

The optimization problem has two parts:

1.  **What to Minimize:**  We minimize `(1/2) * ||w||²`.
    *   `||w||` is the "length" (norm) of a vector `w`. This vector `w` is perpendicular to the separating line/hyperplane.
    *   Minimizing `||w||` is the *same* as maximizing the margin.  They are inversely related:  `margin = 1/||w||`.
    *   We use `||w||²` (the squared length) because it's easier to work with mathematically. The `1/2` is just for convenience in the calculations.

2.  **The Rules (Constraints):**  We have rules that *must* be followed:

    ```
    y_i * (w · x_i + b) ≥ 1   for every data point x_i
    ```

    *   `x_i`:  A data point (like a point on a graph).
    *   `y_i`:  Tells us which class the point `x_i` belongs to (+1 for one class, -1 for the other).
    *   `w`: The same vector as above (perpendicular to the separating line).
    *   `b`:  A "shift" value that moves the line up or down (or shifts the hyperplane).
    *   `w · x_i`: This calculates a value related to how far `x_i` is from the line.
    *   `y_i * (w · x_i + b)`: This makes sure that:
        *   Points are on the *correct* side of the line. If the point is classified correctly, the result will always be positive.
        *   The closest points to the line (the *support vectors*) are a certain distance away (that distance is related to the margin).
   *  `≥ 1` This part of the rule enforces both correct classification *and* that the margin is as big as we've calculated (1/||w||).

**Putting It All Together:**

*   **Minimize:** `(1/2) * ||w||²`  (to maximize the margin)
*   **Subject to:**  `y_i * (w · x_i + b) ≥ 1`  (for all data points – ensures correct classification and defines the margin)

**In Plain English:**

Find the `w` and `b` that make the "street" as wide as possible (by minimizing `||w||²`) *while* making sure *all* the data points are on the correct side of the street and at least a certain distance away from the center line (the constraint `y_i * (w · x_i + b) ≥ 1`). The points that are *exactly* on the edge of the "street" are the support vectors.

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
