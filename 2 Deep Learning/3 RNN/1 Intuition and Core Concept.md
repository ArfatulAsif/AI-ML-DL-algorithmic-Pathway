

## 🔁 What is an RNN?

An **RNN (Recurrent Neural Network)** is a type of neural network designed to handle **sequential data** by maintaining a **hidden state** that captures information from **previous time steps**.

### ✅ Best Suited For:

-   **Time series forecasting** (e.g., stock prices, weather)
    
-   **Natural language processing (NLP)** (e.g., sentiment analysis, text generation)
    
-   **Speech recognition**
    
-   **Video frame analysis**
    
-   **Music generation**
    

----------

## 🧠 How It Works (Intuition)

At each time step tt, an RNN takes:

-   The **input** at that step xtx_t
    
-   The **hidden state** from the previous step ht−1h_{t-1}
    

It computes a new hidden state hth_t, which carries the memory of the sequence so far.

Then it may optionally produce an **output** yty_t.

----------

## 📐 Mathematical Formulation

### Hidden State Update:

ht=tanh⁡(Wxhxt+Whhht−1+bh)h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)

-   xtx_t: input at time tt
    
-   hth_t: hidden state at time tt
    
-   WxhW_{xh}: weights from input to hidden
    
-   WhhW_{hh}: weights from hidden to hidden
    
-   bhb_h: bias
    
-   tanh⁡\tanh: activation function (can be ReLU or others too)
    

### Output (if needed):

yt=softmax(Whyht+by)y_t = \text{softmax}(W_{hy} h_t + b_y)

----------

## 🔁 Sequence Unrolling

For input sequence [x1,x2,...,xT][x_1, x_2, ..., x_T], RNN unrolls like this:

```
x1 --> [h1] --> y1
       ↑
x2 --> [h2] --> y2
       ↑
... and so on

```

Hidden state at each time step depends on all previous inputs.

----------

## 🧮 Backpropagation Through Time (BPTT)

To train RNNs, we use **BPTT**, an extension of backpropagation that unfolds the network in time and applies the chain rule across time steps.

Challenges:

-   **Vanishing gradient**: Gradients shrink too much → long-term dependencies are lost.
    
-   **Exploding gradient**: Gradients grow too large → unstable updates.
    

➡ Solutions: Gradient clipping, better architectures like **LSTM**, **GRU**.

----------

## ⚙️ RNN Variants

### 🌀 12. RNN Variants and Their Purpose

| **Variant**            | **Purpose**                                        |
|------------------------|----------------------------------------------------|
| LSTM (Long Short-Term Memory) | Handles long dependencies using gates             |
| GRU (Gated Recurrent Unit)    | Similar to LSTM, simpler and faster               |
| Bidirectional RNN             | Reads the sequence in both directions            |
| Stacked RNNs                  | Builds depth by stacking multiple RNN layers     |


----------

## 🎛️ Hyperparameters of RNNs

| **Hyperparameter**     | **Description**                                    |
|------------------------|----------------------------------------------------|
| Hidden size            | Size of hidden layer (e.g., 128, 256)             |
| Num layers             | How many RNN layers (stacked RNNs)               |
| Learning rate          | Step size in optimizer                            |
| Batch size             | Number of sequences in a batch                    |
| Sequence length        | Length of input chunks (especially in BPTT)       |
| Dropout                | Regularization to prevent overfitting             |
| Optimizer              | Adam, RMSprop (work well with RNNs)              |
| Activation function    | Usually `tanh` or `ReLU`                          |
| Cell type              | Vanilla RNN / LSTM / GRU                          |


----------

## 🧪 Tuning Hyperparameters

1.  **Start small**: Try small hidden sizes, 1–2 layers.
    
2.  **Monitor overfitting**: Use validation loss & dropout.
    
3.  **Experiment**:
    
    -   Try `hidden_size`: [64, 128, 256]
        
    -   `num_layers`: [1, 2, 3]
        
    -   `learning_rate`: [1e-3, 1e-4]
        
    -   `cell type`: GRU vs LSTM
        
4.  **Use tools**: GridSearchCV (for small data), Optuna, Ray Tune
    

----------

## 📊 Practical Example: Sequence Classification (NLP)

### Task: Sentiment classification on movie reviews (positive/negative)

### Dataset: IMDb (or custom)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Step 1: Load & tokenize
tokenizer = get_tokenizer('basic_english')
train_iter = IMDB(split='train')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Step 2: Define model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

# Step 3: Train
model = RNNModel(len(vocab), embed_dim=100, hidden_dim=128, output_dim=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# You’d also write DataLoader and train/validate functions

```

----------

## 🧠 How RNN Solves Problems

| **Problem**          | **RNN Strength**                              |
|----------------------|-----------------------------------------------|
| Sentiment Analysis   | Understands word sequences                    |
| Stock Prediction     | Uses past trends                              |
| Text Generation      | Remembers previous words                      |
| Speech-to-Text       | Handles audio frames as sequences             |
| Translation          | Learns grammar and context over time          |


----------

## 🧪 Tips for Effective RNN Use

-   Normalize input sequences.
    
-   Use **padding** and **pack_padded_sequence** in PyTorch for batching.
    
-   Use **LSTM/GRU** for complex tasks.
    
-   Combine with **Attention** for better results in long sequences.
    
-   Use **Teacher Forcing** in sequence generation tasks.
    

----------

## 📚 Resources for Further Study

-   **CS231n** – RNN & LSTM lectures
    
-   **PyTorch tutorials** – Text classification, language modeling
    
-   **Karpathy’s blog** – [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
    
-   **Andrej Karpathy's char-RNN code**
 
