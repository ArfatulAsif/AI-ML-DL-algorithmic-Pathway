### 📉 6. Model Fitness / Training Behavior

**What loss function is used?**
The loss function depends on the type of problem:
-   **For Classification**:
    -   `BinaryCrossentropy` for two-class problems.
    -   `CategoricalCrossentropy` or `SparseCategoricalCrossentropy` for multi-class problems.
-   **For Regression**:
    -   `MeanSquaredError` (MSE) or `MeanAbsoluteError` (MAE).

**What can you check:**
- [x] **Learning curves**: Plotting training & validation loss/accuracy over epochs is the primary way to diagnose model behavior.
- [x] **Overfitting signs**: A large and growing gap between training and validation performance (e.g., training loss decreases while validation loss increases).
- [x] **Underfitting clues**: Both training and validation metrics are poor and plateau quickly, indicating the model is too simple to capture the data's patterns.

---
### 📊 7. Evaluation Metrics

**For classification:**
- [x] Accuracy
- [x] Precision / Recall
- [x] F1-Score
- [x] ROC-AUC

**For regression:**
- [x] MAE
- [x] MSE / RMSE
- [x] R²

**Other visuals:**
- [x] Confusion Matrix
- [x] PR Curve
