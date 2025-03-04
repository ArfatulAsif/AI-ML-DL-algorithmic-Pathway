# Decision Tree - Gini, Entropy, and Information Gain Calculation

## Step 1: Understanding Gini Impurity
Gini impurity measures how *pure* or *impure* a dataset is. It is calculated as:

$$
Gini = 1 - \sum_{i=1}^{c} (p_i)^2
$$

Where:  
- \( p_i \) = Proportion of class *i* in the current node.  
- \( c \) = Number of classes (for binary classification, *0 and 1*).  

### Example Calculation:
Given a node with **10 samples**:  
- 6 belong to Class 0  
- 4 belong to Class 1  

The probabilities are:  
- \( p_0 = \frac{6}{10} = 0.6 \)  
- \( p_1 = \frac{4}{10} = 0.4 \)  

Now, calculating Gini:

$$
Gini = 1 - (0.6^2 + 0.4^2)
$$

$$
= 1 - (0.36 + 0.16)
$$

$$
= 1 - 0.52 = 0.48
$$

Thus, the Gini impurity of this node is **0.48**.

---

## Step 2: Entropy Calculation
Entropy measures uncertainty in the dataset. It is given by:

$$
H = - \sum_{i=1}^{c} p_i \log_2(p_i)
$$

For the same example:

$$
H = - (0.6 \log_2 0.6 + 0.4 \log_2 0.4)
$$

Approximating log values:
$$
- \( \log_2(0.6) \approx -0.737 \)
$$

$$
- \( \log_2(0.4) \approx -1.322 \)
$$

Now:

$$
H = - (0.6 \times -0.737 + 0.4 \times -1.322)
$$

$$
= - (-0.442 - 0.529)
$$

$$
= 0.971
$$

Thus, the entropy of this node is **0.971**.

---

## Step 3: Information Gain Calculation
Information Gain (IG) is calculated as:

$$
IG = H_{\text{parent}} - \sum_{j=1}^{k} \frac{N_j}{N} H_j
$$

Where:
- \( H_{\text{parent}} \) = Entropy of the parent node.
- \( k \) = Number of child nodes after a split.
- \( N_j \) = Number of samples in child node \( j \).
- \( N \) = Total samples in the parent node.
- \( H_j \) = Entropy of child node \( j \).

---

## Step 4: Example of Information Gain Calculation
Suppose the parent node has **10 samples**, and after a split:
- Left child: **4 samples** (3 in Class 0, 1 in Class 1)
- Right child: **6 samples** (3 in Class 0, 3 in Class 1)

### Entropy of Left Child:

$$
H_{\text{left}} = - \left( \frac{3}{4} \log_2 \frac{3}{4} + \frac{1}{4} \log_2 \frac{1}{4} \right)
$$

Approximating log values:
- \( \log_2(0.75) \approx -0.415 \)
- \( \log_2(0.25) \approx -2.000 \)

$$
H_{\text{left}} = - (0.75 \times -0.415 + 0.25 \times -2.000)
$$

$$
= - (-0.311 - 0.500)
$$

$$
= 0.811
$$

### Entropy of Right Child:

$$
H_{\text{right}} = - \left( \frac{3}{6} \log_2 \frac{3}{6} + \frac{3}{6} \log_2 \frac{3}{6} \right)
$$

Since both classes are equal, entropy is:

$$
H_{\text{right}} = - (0.5 \log_2 0.5 + 0.5 \log_2 0.5)
$$

$$
= - (0.5 \times -1 + 0.5 \times -1)
$$

$$
= 1.0
$$

### Weighted Average Entropy After Split:

$$
H_{\text{children}} = \frac{4}{10} \times 0.811 + \frac{6}{10} \times 1.0
$$

$$
= 0.3244 + 0.6 = 0.924
$$

### Information Gain Calculation:

$$
IG = H_{\text{parent}} - H_{\text{children}}
$$

$$
= 0.971 - 0.924 = 0.047
$$

Thus, the **Information Gain (IG)** for this split is **0.047**.

---

## Conclusion
- **Gini impurity** measures impurity and is used in CART decision trees.
- **Entropy** measures uncertainty and is used in ID3 and C4.5 decision trees.
- **Information Gain** helps decide the best feature for splitting a node.

