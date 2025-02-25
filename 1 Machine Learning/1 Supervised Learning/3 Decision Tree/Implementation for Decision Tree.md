 
### **Decision Tree Implementation for Classification (Loan Eligibility Example)**  
```python
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import tree

# Sample dataset (Loan Eligibility Prediction)
data = {
    'Income': [50000, 60000, 25000, 40000, 80000, 30000, 100000, 12000, 75000, 90000],
    'Credit_Score': [700, 650, 500, 620, 750, 580, 800, 450, 710, 770],
    'Age': [35, 45, 22, 29, 40, 24, 50, 21, 38, 42],
    'Approved': [1, 1, 0, 0, 1, 0, 1, 0, 1, 1]  # 1 = Loan Approved, 0 = Rejected
}

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Define features (X) and target (y)
X = df[['Income', 'Credit_Score', 'Age']]
y = df['Approved']

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Decision Tree model
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Visualize the Decision Tree
plt.figure(figsize=(10,6))
tree.plot_tree(model, feature_names=X.columns, class_names=['Rejected', 'Approved'], filled=True)
plt.show()
```

---

### **Explanation of Code:**
1. **Dataset Creation:** A simple dataset with features **Income, Credit Score, Age** and target **Approved (1) / Rejected (0)**.  
2. **Splitting Data:** 80% training, 20% testing.  
3. **Model Training:** Using `DecisionTreeClassifier` with Gini Impurity and a depth of 3.  
4. **Predictions & Accuracy:** Evaluates model accuracy.  
5. **Visualization:** Displays a Decision Tree diagram using `plot_tree()`.
