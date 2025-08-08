# 2. **Type of Problems It Solves**

* ☑ **Classification** (e.g., image classification like cat vs. dog)
* ☑ **Regression** (e.g., predicting continuous values like age or price)
* ☐ **Clustering** (Not directly, but can be used for feature extraction in unsupervised learning tasks)
* ☐ **Dimensionality Reduction** (Not directly, but helps in reducing the feature space by learning the most relevant features)
* ☐ **Reinforcement Learning** (Can be used in RL, especially in tasks involving image data, such as game playing or robotic vision)

## **What type of data is it best for?**

### Image Data or 2D data or grid like data.

* **Image data**: Great for processing images, detecting objects, and classifying images.
* **Time-Series Data**: Can be used for sequence-based data like video frames or sensor data.
* **Text (in some cases)**: Can process sequential data or text when combined with other techniques (like RNNs or CNNs for text data).

---

# 🚧 3. **Assumptions & Limitations**

**Does it assume:**

* ☐ **Linearity** (No, CNNs use non-linear activation functions like ReLU to capture complex relationships)
* ☐ **Feature independence** (No, CNNs take advantage of spatial relationships and local correlations in the input)
* ☐ **Normal distribution** (No, CNNs don't assume normal distribution; they work well with a variety of data types)

**Is it sensitive to:**

* ☑ **Outliers** (Yes, CNNs can be sensitive to outliers in training data, which can cause overfitting without proper regularization)
* ☐ **Multicollinearity** (No, CNNs focus on learning local patterns through filters, so multicollinearity is not a major issue)
* ☑ **Scaling** (Yes, CNNs benefit from scaled or normalized data, especially for image data, where pixel values are often normalized to \[0, 1])

**Preprocessing needed:**

* **Image data**: Normalization of pixel values (e.g., scaling to \[0, 1] or \[-1, 1]) and sometimes data augmentation (e.g., flipping, rotation).
* **Text data**: May require embedding or conversion into a 2D format (like a bag-of-words or word embeddings).
* **Time-Series data**: Often requires reshaping into a 2D form or applying techniques like sliding windows to process sequential data.

