# Support Vector Machines (SVM)

Support Vector Machines (SVM) are a powerful, supervised learning algorithm used primarily for classification, but also applicable to regression problems. SVMs are known for their ability to handle high-dimensional data and non-linear relationships effectively, thanks to the "kernel trick." They are a type of discriminative classifier, meaning they explicitly define a decision boundary.

## Hyperplanes and Margins

The core concept of SVM is finding the optimal *hyperplane* that best separates different classes of data. A hyperplane is a decision boundary: a line in 2D, a plane in 3D, and a higher-dimensional equivalent for more features. SVM doesn't just find *any* separating hyperplane; it aims for the one with the *maximum margin*. The margin is the distance between the hyperplane and the closest data points from each class. These closest points are called *support vectors*. A larger margin generally indicates a more robust and generalizable classifier.

**Mathematical Representation:**

In a binary classification problem, let's assume we have training data  {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}, where xᵢ is a feature vector in a d-dimensional space (xᵢ ∈ ℝᵈ), and yᵢ is the class label (+1 or -1).  The goal of SVM is to find a hyperplane defined by:

**w**ᵀ**x** + b = 0

where:
*   **w** is the weight vector (normal to the hyperplane).
*   **x** is the input vector.
*   b is the bias (or intercept) term.

The decision function for classifying a new data point **x** is:

f(**x**) = sign(**w**ᵀ**x** + b)

## Support Vectors

Support vectors are the critical data points that define the margin and, consequently, the optimal hyperplane. They are the points "closest" to the decision boundary. Only these support vectors are needed to define the classifier; other data points further away are less relevant. This makes SVM relatively memory-efficient.

**Mathematical Significance:**

The support vectors are the data points for which the following condition holds:

yᵢ(**w**ᵀ**xᵢ** + b) = 1

This is because they lie exactly on the margin boundaries.

## Non-Linear Separability and the Kernel Trick

Many real-world datasets aren't linearly separable (i.e., you can't draw a straight line/plane to separate the classes). This is where the "kernel trick" comes in. A *kernel* is a function that implicitly transforms the data into a higher-dimensional space where linear separation *is* possible. The SVM algorithm doesn't explicitly perform this transformation; the kernel function calculates the dot products (relationships) between data points *as if* they were in the higher-dimensional space. This avoids the computational cost of actually transforming the data.

**Mathematical Formulation:**

The kernel trick replaces the dot product **xᵢ**ᵀ**xⱼ** in the original space with a kernel function K(**xᵢ**, **xⱼ**):

K(**xᵢ**, **xⱼ**) = φ(**xᵢ**)ᵀφ(**xⱼ**)

where φ(**x**) is a mapping function that transforms **x** from the input space to a higher-dimensional feature space.  We *don't* need to know φ(**x**) explicitly; we only need to know the kernel function K.

## Common Kernel Functions

*   **Linear Kernel:** `K(x, xi) = sum(x * xi)`. Used for linearly separable data. Essentially, it's just a dot product.
*   **Polynomial Kernel:** `K(x,xi) = (gamma * sum(x * xi) + coef0)^d`. Introduces non-linearity; `d` is the degree of the polynomial, `gamma` is a scaling factor, and `coef0` is an independent term.
*   **Radial Basis Function (RBF) Kernel:** `K(x,xi) = exp(-gamma * sum((x – xi)^2))`. A very popular and powerful kernel. `gamma` controls the influence of each data point; a smaller `gamma` gives each point a wider influence. A good default value for gamma is 0.1.
*   **Sigmoid Kernel:** `K(x, xi) = tanh(gamma * sum(x * xi) + coef0)`. Similar to a neural network's activation function.

## Classification Problems

SVMs are primarily used for classification. Given a set of labeled training data (e.g., images of cats and dogs, with each image labeled as "cat" or "dog"), the SVM finds the optimal hyperplane to separate the classes. When a new, unlabeled data point (a new image) is presented, the SVM determines which side of the hyperplane it falls on and assigns the corresponding class label.

## Regression Problems

Although less common, SVMs can also be used for regression. Instead of finding a separating hyperplane, the goal is to find a hyperplane that best fits the data, minimizing the error within a certain margin (similar to a "tube" around the hyperplane).  This is called Support Vector Regression (SVR).

## Training Phase (Optimization Problem)

The training phase involves solving a quadratic programming (QP) optimization problem. The goal is to maximize the margin while minimizing classification errors.

**Primal Form (for linearly separable case):**

Minimize:  (1/2) ||**w**||²
Subject to:  yᵢ(**w**ᵀ**xᵢ** + b) ≥ 1  for all i = 1, ..., n

**Dual Form (more general and allows for kernels):**

Maximize:  ∑ᵢ αᵢ - (1/2) ∑ᵢ∑ⱼ αᵢαⱼyᵢyⱼK(**xᵢ**, **xⱼ**)
Subject to:  ∑ᵢ αᵢyᵢ = 0
            0 ≤ αᵢ ≤ C  for all i = 1, ..., n

where:
*   αᵢ are Lagrange multipliers (one for each data point).
*   C is the regularization parameter (soft margin constant).

The solution to the dual problem gives us the αᵢ values.  The support vectors are the data points where αᵢ > 0.  The weight vector **w** can be expressed in terms of the support vectors and their Lagrange multipliers:

**w** = ∑ᵢ αᵢyᵢ**xᵢ** (sum over support vectors)

The bias term 'b' can be calculated using any support vector and the Karush-Kuhn-Tucker (KKT) conditions.

## An Example:

Imagine a dataset of points in 2D space, representing two different types of flowers (e.g., Iris species) based on sepal length and width. Some points are labeled "Setosa," and others are labeled "Versicolor."

*   **Linearly Separable Case:** If a straight line can clearly separate the Setosa points from the Versicolor points, a linear SVM can be used. The algorithm finds the line that maximizes the margin between the closest points of each class.

*   **Non-Linearly Separable Case:** If the points are arranged in a way that a straight line *cannot* separate them (e.g., a circular arrangement), a kernel trick is needed. An RBF kernel might be used to implicitly map the data to a higher dimension where a hyperplane *can* separate the classes.

Once the SVM is trained (either with a linear or non-linear kernel), a new flower with its sepal length and width can be classified by determining which side of the decision boundary it falls on.

## Hyperparameter Tuning

*   **C (Regularization):** Controls the trade-off between maximizing the margin and minimizing classification errors on the *training* data. A smaller C allows for a wider margin but may misclassify more training points (soft margin). A larger C forces a smaller margin but tries to classify training points more accurately (hard margin).  Mathematically, C adds a penalty for misclassified points in the optimization problem.
*   **Kernel:** Choosing the appropriate kernel (linear, polynomial, RBF, etc.) is crucial and depends on the data.
*   **Gamma (for RBF kernel):** Controls the influence of each data point. Smaller gamma = wider influence; larger gamma = tighter influence (can lead to overfitting).  Mathematically, it's the inverse of the radius of influence of the support vectors.
*    **Degree(d) (for polynomial kernel):** specify the d.
* **coef0 (for polynomial and sigmoid kernels):**  An independent term that influences the model's flexibility.

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
