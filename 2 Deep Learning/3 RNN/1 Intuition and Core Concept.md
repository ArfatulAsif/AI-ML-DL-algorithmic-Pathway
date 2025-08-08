
## 🔁 What is an RNN?

An **RNN (Recurrent Neural Network)** is a type of neural network designed to handle **sequential data** by maintaining a **hidden state** that captures information from **previous time steps**.

### ✅ Best Suited For:

* **Time series forecasting** (e.g., stock prices, weather)
* **Natural language processing (NLP)** (e.g., sentiment analysis, text generation)
* **Speech recognition**
* **Video frame analysis**
* **Music generation**

---

## 🧠 How It Works (Intuition)

At each time step \$t\$, an RNN takes:

* The **input** at that step \$x\_t\$
* The **hidden state** from the previous step \$h\_{t-1}\$

It computes a new hidden state \$h\_t\$, which carries the memory of the sequence so far.

Then it may optionally produce an **output** \$y\_t\$.

---

## 📐 Mathematical Formulation

### Hidden State Update:

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

Where:

* \$x\_t\$: input at time \$t\$
* \$h\_t\$: hidden state at time \$t\$
* \$W\_{xh}\$: weights from input to hidden
* \$W\_{hh}\$: weights from hidden to hidden
* \$b\_h\$: bias
* \$\tanh\$: activation function (can be ReLU or others too)

### Output (if needed):

$$
y_t = \text{softmax}(W_{hy} h_t + b_y)
$$


---

## 🧮 Backpropagation Through Time (BPTT)

To train RNNs, we use **BPTT**, an extension of backpropagation that unfolds the network in time and applies the chain rule across time steps.

**Challenges**:

* **Vanishing gradient**: Gradients shrink too much → long-term dependencies are lost.
* **Exploding gradient**: Gradients grow too large → unstable updates.

➡ **Solutions**: Gradient clipping, better architectures like **LSTM**, **GRU**.

---

## ⚙️ RNN Variants

### 🌀 12. RNN Variants and Their Purpose

| **Variant**                   | **Purpose**                                  |
| ----------------------------- | -------------------------------------------- |
| LSTM (Long Short-Term Memory) | Handles long dependencies using gates        |
| GRU (Gated Recurrent Unit)    | Similar to LSTM, simpler and faster          |
| Bidirectional RNN             | Reads the sequence in both directions        |
| Stacked RNNs                  | Builds depth by stacking multiple RNN layers |

---

## 🎛️ Hyperparameters of RNNs

| **Hyperparameter**  | **Description**                             |
| ------------------- | ------------------------------------------- |
| Hidden size         | Size of hidden layer (e.g., 128, 256)       |
| Num layers          | How many RNN layers (stacked RNNs)          |
| Learning rate       | Step size in optimizer                      |
| Batch size          | Number of sequences in a batch              |
| Sequence length     | Length of input chunks (especially in BPTT) |
| Dropout             | Regularization to prevent overfitting       |
| Optimizer           | Adam, RMSprop (work well with RNNs)         |
| Activation function | Usually `tanh` or `ReLU`                    |
| Cell type           | Vanilla RNN / LSTM / GRU                    |

---

## 🧪 Tuning Hyperparameters

1. **Start small**: Try small hidden sizes, 1–2 layers.
2. **Monitor overfitting**: Use validation loss & dropout.
3. **Experiment**:

   * `hidden_size`: \[64, 128, 256]
   * `num_layers`: \[1, 2, 3]
   * `learning_rate`: \[1e-3, 1e-4]
   * `cell type`: GRU vs LSTM
4. **Use tools**: GridSearchCV (for small data), Optuna, Ray Tune

---

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
```

---

## 🧠 How RNN Solves Problems

| **Problem**        | **RNN Strength**                     |
| ------------------ | ------------------------------------ |
| Sentiment Analysis | Understands word sequences           |
| Stock Prediction   | Uses past trends                     |
| Text Generation    | Remembers previous words             |
| Speech-to-Text     | Handles audio frames as sequences    |
| Translation        | Learns grammar and context over time |

---

## 🧪 Tips for Effective RNN Use

* Normalize input sequences.
* Use **padding** and `pack_padded_sequence` in PyTorch for batching.
* Use **LSTM/GRU** for complex tasks.
* Combine with **Attention** for better results in long sequences.
* Use **Teacher Forcing** in sequence generation tasks.

---

Great! Let's understand **RNN** in the **simplest way**—like your favorite teacher sitting beside you with a whiteboard.

---




# For getting a better understanding of RNN, Look at this:


## 🧠 What Is RNN?

An **RNN (Recurrent Neural Network)** is a neural network that learns **patterns in sequences** (like text, time series, etc.) by **remembering past information** using a **hidden state**.

Think of it like this:
🧠 "Hey! I remember what I read before, so I can better understand what's next."

---

## 🎯 Example Dataset: Name Classification

Let’s say we have a **dataset of names** and their **language of origin**:

| Name    | Language |
| ------- | -------- |
| Ivan    | Russian  |
| Pierre  | French   |
| Hiroshi | Japanese |

We want to **predict the language** of a name character by character.

---

## 📥 Step-by-Step RNN Working

Let’s use the name:

```
“Ivan”
```

### ✅ Input Sequence:

We break “Ivan” into characters:
**x₁ = 'I'**, **x₂ = 'v'**, **x₃ = 'a'**, **x₄ = 'n'**

Each character is converted to a **vector** using one-hot encoding or embeddings.
Suppose:

```
x₁ = [1, 0, 0, ..., 0]  # 'I'
x₂ = [0, 1, 0, ..., 0]  # 'v'
...
```

### 🔁 RNN Flow:

At **each time step t**, RNN performs:

```
hₜ = tanh(Wₓₕ·xₜ + Wₕₕ·hₜ₋₁ + bₕ)
```

* **xₜ** = current character input
* **hₜ₋₁** = hidden state from previous step
* **hₜ** = current memory
* **W** and **b** = weights and biases (learned)

Then it outputs:

```
yₜ = softmax(Wₕy·hₜ + bᵧ)
```

* **yₜ** = prediction (maybe of next char or final language class)

---

## 📦 Unrolling Ivan through RNN

Let’s unroll "Ivan":

```
Step 1: x₁ = 'I'
        h₀ = 0 (initial hidden state)
        h₁ = tanh(Wₓₕ·x₁ + Wₕₕ·h₀ + b)
        y₁ = output (not final prediction)

Step 2: x₂ = 'v'
        h₂ = tanh(Wₓₕ·x₂ + Wₕₕ·h₁ + b)
        y₂ = output

Step 3: x₃ = 'a'
        h₃ = tanh(Wₓₕ·x₃ + Wₕₕ·h₂ + b)
        y₃ = output

Step 4: x₄ = 'n'
        h₄ = tanh(Wₓₕ·x₄ + Wₕₕ·h₃ + b)
        y₄ = final output → classify as “Russian”
```

👉 **Final hidden state h₄** is passed to a classifier (like a linear layer + softmax) to predict the **language**.

---

## 🎨 Visual Intuition

```
'I' → [h₁] →
'v' → [h₂] →
'a' → [h₃] →
'n' → [h₄] → Final Output: Russian
```

Each **hidden state** remembers what came before.

---

## 💡 Analogy

Imagine reading the word "Ivan" one letter at a time.

* At 'I', you know very little.
* At 'v', you’re thinking: "Sounds like a Slavic name..."
* At 'a', your brain gets more confident.
* At 'n', your memory connects all letters: “Yep! That’s Russian.”

This is exactly how RNN learns.

---

## 📌 Summary of Core Components

| Concept    | Meaning                                                            |
| ---------- | ------------------------------------------------------------------ |
| `xₜ`       | Input at time step t (e.g., a letter like 'v')                     |
| `hₜ`       | Memory that keeps updating (contains knowledge of previous inputs) |
| `yₜ`       | Output (can be at each step or only at the end)                    |
| Weights    | Learns how to combine inputs + memory                              |
| Activation | `tanh` or `ReLU` to add non-linearity                              |

---

## 🧪 Mini Practice Question (For You!)

Say the word is "Hiroshi" — can you write the input steps?
How many characters?
How many hidden states?

Try answering it like this:

```
Step 1: x₁ = 'H' → h₁
Step 2: x₂ = 'i' → h₂
...
Final output: Japanese
```

