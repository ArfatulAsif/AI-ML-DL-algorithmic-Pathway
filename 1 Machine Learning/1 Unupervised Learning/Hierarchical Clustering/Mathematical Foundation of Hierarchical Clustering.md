### **Mathematical Foundation of Hierarchical Clustering**
Hierarchical clustering groups data points based on similarity by iteratively merging or splitting clusters. It primarily relies on two key mathematical components:  
1. **Distance Metrics** (measuring similarity between data points).  
2. **Linkage Criteria** (determining how clusters are merged).  

<br>

---

<br>


## **1. Distance Metrics (Measuring Similarity)**
Distance metrics are used to determine how close two data points (or clusters) are.  

### **1.1 Euclidean Distance**
The most common distance metric is **Euclidean Distance**, which measures the straight-line distance between two points.

For two points:  

$$
A(x_1, y_1), B(x_2, y_2)
$$

The Euclidean distance is given by:  

$$
d(A, B) = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
$$

For **n-dimensional** data points \( A(x_{1A}, x_{2A}, ..., x_{nA}) \) and \( B(x_{1B}, x_{2B}, ..., x_{nB}) \), the formula extends to:

$$
d(A, B) = \sqrt{\sum_{i=1}^{n} (x_{iB} - x_{iA})^2}
$$

### **1.2 Manhattan Distance (City Block Distance)**
Instead of measuring the direct straight-line distance, **Manhattan Distance** (or L1 norm) measures the sum of the absolute differences between coordinates.

$$
d(A, B) = \sum_{i=1}^{n} |x_{iB} - x_{iA}|
$$

This is often used when movement is restricted to a grid (e.g., taxi movement in a city).  

### **1.3 Cosine Similarity**
Rather than distance, **Cosine Similarity** measures the angle between two vectors \( A \) and \( B \):

$$
\cos(\theta) = \frac{A \cdot B}{||A|| \cdot ||B||}
$$

Where:
- \( A \cdot B \) is the **dot product** of vectors \( A \) and \( B \).
- \( ||A|| \) and \( ||B|| \) are the **magnitudes** (Euclidean norms).

Cosine similarity is commonly used in **text mining and NLP**, where documents are represented as word frequency vectors.

<br>

---

<br>


## **2. Linkage Criteria (Determining How Clusters Are Merged)**
Once distances are computed, hierarchical clustering determines how to merge clusters using **linkage criteria**.

### **2.1 Single Linkage (Minimum Distance)**

$$
d(A, B) = \min \{ d(x_i, x_j) \}, \quad x_i \in A, x_j \in B
$$

- Merges clusters based on **the closest pair of points**.  
- Forms **long, chain-like clusters** (can lead to a chaining effect).  
- **Example**: If clusters A and B have points:
  - \( A = \{(1,1), (2,2)\} \), \( B = \{(8,8), (9,9)\} \)
  - The minimum distance might be between **(2,2) and (8,8)**.

### **2.2 Complete Linkage (Maximum Distance)**

$$
d(A, B) = \max \{ d(x_i, x_j) \}, \quad x_i \in A, x_j \in B
$$

- Merges clusters based on **the farthest pair of points**.  
- Tends to create **compact, well-separated clusters**.  
- **Example**: Using the same clusters, the distance might be measured between **(1,1) and (9,9)**.

### **2.3 Average Linkage**

$$
d(A, B) = \frac{1}{|A| |B|} \sum_{x_i \in A} \sum_{x_j \in B} d(x_i, x_j)
$$

- Uses the **average distance** between all points of the two clusters.
- A compromise between **Single and Complete Linkage**.

### **2.4 Centroid Linkage**

$$
d(A, B) = || C_A - C_B ||
$$

Where:

$$
C_A = \frac{1}{|A|} \sum_{x \in A} x, \quad C_B = \frac{1}{|B|} \sum_{x \in B} x
$$

- Merges clusters based on **centroids (mean points)**.  
- Used in **UPGMA (Unweighted Pair Group Method with Arithmetic Mean)**.


<br>

---

<br>

## Example with Proximity Matrix
Let's apply **Agglomerative Hierarchical Clustering** step-by-step on the dataset:  

$$
X = \{1, 5, 8, 10, 19, 20\}
$$

### **Step 1: Compute Distance Matrix**

|   | **1** | **5** | **8** | **10** | **19** | **20** |
|---|---|---|---|---|---|---|
| **1**  | 0  | 4  | 7  | 9  | 18  | 19  |
| **5**  | 4  | 0  | 3  | 5  | 14  | 15  |
| **8**  | 7  | 3  | 0  | 2  | 11  | 12  |
| **10** | 9  | 5  | 2  | 0  | 9   | 10  |
| **19** | 18 | 14 | 11 | 9  | 0   | 1   |
| **20** | 19 | 15 | 12 | 10 | 1   | 0   |


### **Step 2: Merge Closest Clusters**
1. **(19,20) → Cluster A**
2. **(8,10) → Cluster B**
3. **(5,8,10) → Cluster C**
4. **(1,5,8,10) → Cluster D**
5. **(D, A) → Final Cluster**

---
