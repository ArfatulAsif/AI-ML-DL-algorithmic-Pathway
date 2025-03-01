
### **Step 1: Understanding Gini Impurity**
Gini impurity measures how **pure** or **impure** a dataset is. It is calculated as:

\[
Gini = 1 - \sum_{i=1}^{c} (p_i)^2
\]

Where:
- \( p_i \) = Proportion of class **i** in the current node.
- \( c \) = Number of classes (for binary classification, **0 and 1**).

---

### **Step 2: Sample Dataset**
Our dataset contains **three features**: **Income, Credit Score, and Age**, with a target variable **Approved (1) / Rejected (0)**.

| Income | Credit Score | Age | Approved (Target) |
|--------|-------------|-----|------------------|
| 50000  | 700         | 35  | 1 |
| 60000  | 650         | 45  | 1 |
| 25000  | 500         | 22  | 0 |
| 40000  | 620         | 29  | 0 |
| 80000  | 750         | 40  | 1 |
| 30000  | 580         | 24  | 0 |
| 100000 | 800         | 50  | 1 |
| 12000  | 450         | 21  | 0 |
| 75000  | 710         | 38  | 1 |
| 90000  | 770         | 42  | 1 |

Total samples: **10**  
Loan Approved (1): **6**  
Loan Rejected (0): **4**

---

### **Step 3: Compute Gini Impurity for Root Node**
At the root node, we have:

\[
Gini = 1 - (p_1^2 + p_0^2)
\]

Where:
- \( p_1 = \frac{6}{10} = 0.6 \) (Loan Approved)
- \( p_0 = \frac{4}{10} = 0.4 \) (Loan Rejected)

\[
Gini = 1 - (0.6^2 + 0.4^2)
\]

\[
= 1 - (0.36 + 0.16) = 1 - 0.52 = 0.48
\]

**Root Gini Impurity: 0.48**

---

### **Step 4: Splitting on "Income" Feature**
Let's assume we split on **Income > 50000**. The dataset is divided into two branches:

#### **Left Node (Income ≤ 50000)**:
| Income | Approved |
|--------|----------|
| 25000  | 0 |
| 40000  | 0 |
| 30000  | 0 |
| 12000  | 0 |

- 4 samples → **All Rejected (0)**
- \( p_1 = 0 \), \( p_0 = 1 \)

\[
Gini = 1 - (0^2 + 1^2) = 1 - (0 + 1) = 0
\]

**Gini Impurity of Left Node = 0 (Pure Node)**

---

#### **Right Node (Income > 50000)**:
| Income | Approved |
|--------|----------|
| 50000  | 1 |
| 60000  | 1 |
| 80000  | 1 |
| 100000 | 1 |
| 75000  | 1 |
| 90000  | 1 |

- 6 samples → **All Approved (1)**
- \( p_1 = 1 \), \( p_0 = 0 \)

\[
Gini = 1 - (1^2 + 0^2) = 1 - (1 + 0) = 0
\]

**Gini Impurity of Right Node = 0 (Pure Node)**

---

### **Step 5: Compute Weighted Gini Impurity After Split**
\[
Gini_{new} = \frac{4}{10} \times 0 + \frac{6}{10} \times 0
\]

\[
= 0 + 0 = 0
\]

Since the new Gini is **0 (pure nodes)**, this is a **perfect split** using **Income**.

---

### **Entropy Calculation (If We Use Entropy Instead of Gini)**
Entropy is another metric used for decision trees (used in "Information Gain"). It is calculated as:

\[
Entropy = - \sum_{i=1}^{c} p_i \log_2 p_i
\]

At the **root node**:
\[
Entropy = - (0.6 \log_2 0.6 + 0.4 \log_2 0.4)
\]

\[
= - (0.6 \times -0.736) - (0.4 \times -1.322)
\]

\[
= 0.441 + 0.528 = 0.971
\]

If we split **Income ≤ 50000** and **Income > 50000**, the new entropy would be:

\[
Entropy_{new} = \frac{4}{10} \times 0 + \frac{6}{10} \times 0 = 0
\]

The **Information Gain** is:

\[
IG = Entropy_{root} - Entropy_{new} = 0.971 - 0 = 0.971
\]

This means splitting on **Income** is a **good decision** based on both Gini and Entropy.

---

### **Conclusion**
- **Before Splitting:** Gini = **0.48**, Entropy = **0.971**.
- **After Splitting on Income > 50000:** Gini = **0**, Entropy = **0** (Pure nodes).
- This proves **Income** is a strong decision factor for loan approval.

Would you like a code implementation of these calculations? 🚀
