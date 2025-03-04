### **Step 1: Understanding Gini Impurity**  
Gini impurity measures how **pure** or **impure** a dataset is. It is calculated as:  

$$
\text{Gini} = 1 - \sum_{i=1}^{c} (p_i)^2
$$

Where:  
- \( p_i \) = Proportion of class **i** in the current node.  
- \( c \) = Number of classes (for binary classification, **0 and 1**).  

#### **Example Calculation of Gini Impurity**  
Suppose we have a node containing 10 samples:  
- 6 belong to Class 0  
- 4 belong to Class 1  

The probabilities of each class are:  
- \( p_0 = \frac{6}{10} = 0.6 \)  
- \( p_1 = \frac{4}{10} = 0.4 \)  

Now, applying the Gini formula:  

$$
\text{Gini} = 1 - (0.6^2 + 0.4^2)
$$

$$
= 1 - (0.36 + 0.16)
$$

$$
= 1 - 0.52 = 0.48
$$

Thus, the Gini impurity of this node is **0.48**.

---

### **Step 2: Understanding Entropy (Information Gain)**  
Entropy is a measure of randomness or uncertainty in a dataset. It is given by:  

$$
H = - \sum_{i=1}^{c} p_i \log_2(p_i)
$$

Where:  
- \( p_i \) = Proportion of class **i** in the current node.  
- \( c \) = Number of classes.  

#### **Example Calculation of Entropy**  
Using the same example as above:  

$$
H = - (0.6 \log_2(0.6) + 0.4 \log_2(0.4))
$$

Using approximate log values:  
- \( \log_2(0.6) \approx -0.737 \)  
- \( \log_2(0.4) \approx -1.322 \)  

Now, calculating:  

$$
H = - (0.6 \times (-0.737) + 0.4 \times (-1.322))
$$

$$
= - ( -0.442 + -0.529 )
$$

$$
= 0.971
$$

Thus, the entropy of this node is **0.971**.

---

### **Step 3: Information Gain (IG) Calculation**  
Information Gain measures how much uncertainty is reduced after splitting a node. It is calculated as:

$$
IG = H_{\text{parent}} - \sum_{j=1}^{k} \frac{N_j}{N} H_j
$$

Where:  
- \( H_{\text{parent}} \) = Entropy of the parent node.  
- \( k \) = Number of child nodes after a split.  
- \( N_j \) = Number of samples in child node \( j \).  
- \( N \) = Total samples in the parent node.  
- \( H_j \) = Entropy of child node \( j \).  

#### **Example Calculation of Information Gain**  
Suppose after a split, we have:  
- **Left Child Node** (4 samples: 3 from Class 0, 1 from Class 1)  
- **Right Child Node** (6 samples: 3 from Class 0, 3 from Class 1)  

First, calculate entropies for both child nodes:  

**Left Node Entropy:**
$$
H_L = - \left( \frac{3}{4} \log_2 \frac{3}{4} + \frac{1}{4} \log_2 \frac{1}{4} \right)
$$

Using approximations:  
- \( \log_2(0.75) \approx -0.415 \)  
- \( \log_2(0.25) \approx -2 \)  

$$
H_L = - (0.75 \times -0.415 + 0.25 \times -2)
$$

$$
= 0.811
$$

**Right Node Entropy:**
$$
H_R = - \left( \frac{3}{6} \log_2 \frac{3}{6} + \frac{3}{6} \log_2 \frac{3}{6} \right)
$$

Since both probabilities are equal (0.5), entropy simplifies to:

$$
H_R = - (0.5 \times -1 + 0.5 \times -1) = 1
$$

Now, calculate Information Gain:

$$
IG = H_{\text{parent}} - \left( \frac{4}{10} \times 0.811 + \frac{6}{10} \times 1 \right)
$$

$$
= 0.971 - (0.4 \times 0.811 + 0.6 \times 1)
$$

$$
= 0.971 - (0.3244 + 0.6)
$$

$$
= 0.971 - 0.9244 = 0.0466
$$

Thus, the Information Gain for this split is **0.0466**.

---

### **Step 4: Choosing the Best Split**  
To decide which feature to split on, we compare the **Information Gain** for different splits and choose the one with the **highest IG**. The higher the IG, the better the split, as it reduces uncertainty the most.

---

