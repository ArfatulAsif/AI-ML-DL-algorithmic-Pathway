

### 📉 6. Model Fitness / Training Behavior (RNN)

**What loss function is used?**

-   ✅ **`BinaryCrossentropy`**: Used for binary sentiment classification (positive/negative). It measures the difference between predicted probabilities and actual class labels.
    

**What can you check:**

-   **Learning curves**:
    
    -   The script plots **training vs validation loss** over epochs.
        
    -   Helps monitor optimization behavior.
        
-   **Overfitting signs**:
    
    -   Training loss goes down while validation loss starts increasing.
        
    -   Solution: increase dropout, reduce model complexity, early stopping.
        
-   **Underfitting clues**:
    
    -   Both training and validation loss remain high or flat.
        
    -   Indicates model may be too simple or learning rate is too low.
        

----------

### 📊 7. Evaluation Metrics (RNN Classification)

**For binary classification (sentiment analysis):**

-   **Accuracy**: Overall percentage of correct predictions.
    
-   **Precision / Recall / F1-Score**:
    
    -   Precision: How many predicted positives were actually positive.
        
    -   Recall: How many actual positives were predicted correctly.
        
    -   F1-Score: Harmonic mean of precision and recall (balanced metric).
        
-   **Confusion Matrix**: Shown as a heatmap for visual analysis of TP, FP, TN, FN.
    

**Visuals provided:**

-   **Loss curve**: Training & validation loss.
    
-   **Confusion matrix**: Colored with counts of predicted vs actual.
    

