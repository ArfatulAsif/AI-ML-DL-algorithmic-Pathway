


# 🧠 2. Type of Problems It Solves

-   **Classification** (e.g., sentiment analysis, POS tagging)
    
-   **Regression** (e.g., stock price prediction, forecasting)
    
-   Clustering
    
-   Dimensionality Reduction
    
-   Reinforcement Learning _(but RNNs are often used as components in RL, e.g., in recurrent policy networks)_
    

**What type of data is it best for?**  
RNNs are ideal for **sequential or temporal data**, such as:

-   Time series (e.g., sensor data, stock prices)
    
-   Natural Language (e.g., text, speech)
    
-   Audio signals
    
-   Video frame sequences
    

They work best when **order matters**, and there's potential for **dependencies over time**.

----------

# 🚧 3. Assumptions & Limitations

**Does it assume:**

-   Linearity
    
-   Feature independence
    
-   Normal distribution
    

✅ RNNs make **no strong assumptions** about the statistical nature of the data — making them suitable for real-world messy, unstructured, or noisy sequences.

----------

**Is it sensitive to:**

-   **Outliers**: Yes. Outliers in sequences (e.g., sudden spikes in time series) can cause unstable gradient flows, especially during BPTT.
    
-   **Multicollinearity**: RNNs can **absorb correlated features**, but excessive redundancy might slow learning or cause overfitting.
    
-   **Scaling**: **Very sensitive.** Feature scaling is essential for stable and efficient training due to the recurrent nature of the computations.
    

----------

## **Preprocessing Needed?**

Yes — especially important for **temporal data**:

-   📏 **Scaling numerical features** (e.g., with `MinMaxScaler` or `StandardScaler`) is **critical** to prevent exploding or vanishing gradients.
    
-   🔢 **Encoding categorical sequences** (e.g., characters, words, tokens) is done using:
    
    -   `OneHotEncoder` for small vocabularies
        
    -   `Embedding` layers for dense vector representations
        
-   ⏳ **Sequence Padding**: For variable-length sequences, use padding and masking to ensure equal lengths across batches.
    
-   🧼 **Noise removal & smoothing** (for time series or audio) can also improve performance.
    

