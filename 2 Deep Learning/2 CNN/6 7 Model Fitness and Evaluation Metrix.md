# 📉 **6. Model Fitness / Training Behavior**

**What loss function is used?**

* For **classification tasks**: Cross-Entropy Loss (e.g., binary cross-entropy for binary classification or categorical cross-entropy for multi-class classification).
* For **regression tasks**: Mean Squared Error (MSE) or Mean Absolute Error (MAE).

**What can you check?**

* ☑ **Learning curves**: Monitor the training and validation loss over time to assess if the model is improving and converging.
* ☑ **Overfitting signs**: If the training loss keeps decreasing but the validation loss starts to increase, it might indicate overfitting. This suggests that the model is memorizing the training data rather than generalizing.
* ☑ **Underfitting clues**: If both training and validation losses are high, the model is not learning effectively and may be too simple to capture the underlying patterns (underfitting).

---

# 📊 **7. Evaluation Metrics**

For **classification**:

* ☑ **Accuracy**: Measures the percentage of correct predictions out of all predictions.
* ☑ **Precision / Recall**: Useful when dealing with imbalanced datasets. Precision measures the accuracy of positive predictions, and recall measures the ability to capture positive cases.
* ☑ **F1-Score**: The harmonic mean of precision and recall, especially useful when you need a balance between precision and recall.
* ☑ **ROC-AUC**: Measures the area under the Receiver Operating Characteristic curve, which evaluates the trade-off between true positive rate and false positive rate at various thresholds.

For **regression**:

* ☑ **MAE (Mean Absolute Error)**: Measures the average magnitude of errors in a set of predictions, without considering their direction.
* ☑ **MSE (Mean Squared Error) / RMSE (Root Mean Squared Error)**: Measures the average of the squared differences between predicted and actual values. RMSE is the square root of MSE, which gives errors in the same unit as the target variable.
* ☑ **R² (R-squared)**: Indicates how well the regression model explains the variance in the target variable (between 0 and 1, where 1 is perfect fit).

Other visuals:

* ☑ **Confusion Matrix**: A matrix that shows the count of true positive, true negative, false positive, and false negative predictions, helping to understand the performance of the classification model.
* ☑ **PR Curve (Precision-Recall Curve)**: A curve that shows the trade-off between precision and recall for different threshold values, useful for evaluating imbalanced classification tasks.

---
