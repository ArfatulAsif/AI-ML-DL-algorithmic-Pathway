# **Implementation of Hierarchical Clustering**

The following code performs **Hierarchical Clustering** using **single linkage** and visualizes the **dendrogram**.

```python
import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Define data points
data_points = np.array([[1], [5], [8], [10], [19], [20]])

# Compute the pairwise distance matrix (Euclidean by default)
distance_matrix = squareform(pdist(data_points, metric='euclidean'))

print("Distance Matrix:\n", distance_matrix)  

# Perform Agglomerative Hierarchical Clustering
linkage_matrix = sch.linkage(pdist(data_points, metric='euclidean'), method='single')  # Change method if needed

# Plot the dendrogram
plt.figure(figsize=(8, 5))
sch.dendrogram(linkage_matrix, labels=["1", "5", "8", "10", "19", "20"])
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
```

<br>

---

<br>

## **How It Works**

<br>


### **Step 1: Compute Distance Matrix**
The **Euclidean distance** between all pairs of points is computed using:
```python
pdist(data_points, metric='euclidean')
```
This generates the following distance matrix:
```
[[ 0.  4.  7.  9. 18. 19.]
 [ 4.  0.  3.  5. 14. 15.]
 [ 7.  3.  0.  2. 11. 12.]
 [ 9.  5.  2.  0.  9. 10.]
 [18. 14. 11.  9.  0.  1.]
 [19. 15. 12. 10.  1.  0.]]
```

<br>


<br>

### **Step 2: Hierarchical Clustering**
Clusters are merged iteratively based on the **Single Linkage** method (minimum distance between clusters).
```python
linkage_matrix = sch.linkage(pdist(data_points, metric='euclidean'), method='single')
```

<br>



<br>

### **Step 3: Dendrogram Visualization**
The **dendrogram** shows how clusters are formed step by step.
```python
plt.figure(figsize=(8, 5))
sch.dendrogram(linkage_matrix, labels=["1", "5", "8", "10", "19", "20"])
plt.show()
```

<br>

<img src="/pic1.png"  width="450">


<br>

---

<br>


