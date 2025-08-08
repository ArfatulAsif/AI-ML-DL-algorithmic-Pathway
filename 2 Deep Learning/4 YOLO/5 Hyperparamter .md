# ⚙️ YOLO Hyperparameters: Model Architecture & Training Tuning Guide

YOLO (You Only Look Once) is a powerful real-time object detection algorithm. Unlike FNN, which predicts a single label or value per input, YOLO predicts **multiple bounding boxes**, **objectness scores**, and **class probabilities** — all at once, across the entire image. That makes tuning its hyperparameters both more challenging and more exciting.

---

## 🔧 Model Architecture Hyperparameters

These define how the **YOLO model is built structurally** — from the backbone to the detection head.

---

### 1. **Backbone Network**

**Purpose:** Extracts deep features from the image.

* 🔹 **Options:** `Darknet53`, `CSPDarknet`, `MobileNet`, `ResNet`, etc.
* 🔹 **Hyperparameter:** `'backbone': trial.suggest_categorical([...])`
* 🔹 **Tradeoff:**

  * **Lightweight** backbones (e.g., MobileNet) = Faster but less accurate
  * **Deeper** backbones (e.g., CSPDarknet) = More accurate but slower

---

### 2. **Input Image Size**

**Purpose:** Controls the resolution of the input image.

* 🔹 **Typical values:** 320×320, 416×416, 608×608
* 🔹 **Hyperparameter:** `'input_size': trial.suggest_categorical([320, 416, 608])`
* 🔹 **Tradeoff:**

  * Larger images = **better small-object detection** but slower
  * Smaller images = faster, less precise

---

### 3. **Anchor Boxes (Pre-defined box shapes)**

YOLO uses a fixed set of **anchor box dimensions** to “guess” object shapes.

* 🔹 **Hyperparameters:**

  * Number of anchor boxes (e.g., 3, 5, 9)
  * Anchor box sizes (width × height)
* 🔹 Tip: Use **k-means clustering** on your dataset’s bounding boxes to generate better anchors.

---

### 4. **Number of Detection Heads / Feature Scales**

**YOLOv3/v4/v5** support **multi-scale detection** (e.g., 3 scales).

* 🔹 You can tune:

  * Number of scales (e.g., 1, 2, or 3)
  * Feature layers to attach the head (e.g., P3, P4, P5)
* 🔹 More scales = better for **small + large** objects

---

### 5. **Number of Classes**

* Not really a tunable hyperparameter, but must match your dataset.

---

### 6. **Activation Function**

* Typically **Leaky ReLU**, **Mish**, or **Swish**
* `'activation': trial.suggest_categorical(['leaky_relu', 'mish', 'relu'])`

---

## 🏋️ Training Process Hyperparameters

These control **how the YOLO model learns**.

---

### 1. **Optimizer**

* Options: `'adam'`, `'sgd'`, `'rmsprop'`, `'adamW'`
* `'optimizer': trial.suggest_categorical([...])`

---

### 2. **Learning Rate**

* Critical hyperparameter. Should use log-scale sampling.
* `'learning_rate': trial.suggest_float(1e-5, 1e-2, log=True)`
* Warm-up learning rate can also be tuned.

---

### 3. **Batch Size**

* Depends on memory
* Common: 16, 32, 64

---

### 4. **Epochs**

* Choose early stopping or a high value (e.g., 100–300)
* `'epochs': trial.suggest_int(50, 300)`

---

### 5. **IOU Threshold (for NMS)**

* Controls **non-maximum suppression**
* `'iou_threshold': trial.suggest_float(0.4, 0.7)`
* Lower = more boxes (less suppression), higher = more aggressive

---

### 6. **Objectness Threshold**

* Threshold to decide if a predicted box contains an object
* `'objectness_threshold': trial.suggest_float(0.1, 0.5)`

---

### 7. **Loss Weights**

YOLO combines 3 losses:

* 🔹 **Localization Loss (coord)** → position accuracy
* 🔹 **Confidence Loss (obj/noobj)** → object detection certainty
* 🔹 **Classification Loss (cls)** → class prediction

You can tune their **relative importance**:

```python
'lambda_coord': trial.suggest_float(1, 10),
'lambda_noobj': trial.suggest_float(0.1, 1),
'lambda_cls': trial.suggest_float(1, 10)
```

---

## ✨ Optional but Helpful Hyperparameters

| Hyperparameter        | Why It’s Useful                              |
| --------------------- | -------------------------------------------- |
| `data_augmentation`   | Boosts generalization                        |
| `weight_decay`        | Regularizes the model                        |
| `gradient_clipping`   | Prevents exploding gradients                 |
| `cosine_annealing_lr` | Learning rate scheduling for smooth descent  |
| `mixup`, `mosaic`     | Advanced data augmentation (used in YOLOv5+) |

---

## 🧪 Example Optuna Objective Function (YOLO)

```python
def objective(trial):
    params = {
        'backbone': trial.suggest_categorical(['CSPDarknet', 'MobileNet', 'ResNet50']),
        'input_size': trial.suggest_categorical([320, 416, 608]),
        'activation': trial.suggest_categorical(['leaky_relu', 'mish']),
        'learning_rate': trial.suggest_float(1e-5, 1e-2, log=True),
        'optimizer': trial.suggest_categorical(['adam', 'sgd']),
        'batch_size': trial.suggest_categorical([16, 32]),
        'epochs': trial.suggest_int(50, 150),
        'iou_threshold': trial.suggest_float(0.4, 0.7),
        'objectness_threshold': trial.suggest_float(0.1, 0.5),
        'lambda_coord': trial.suggest_float(1, 10),
        'lambda_cls': trial.suggest_float(1, 10),
        'lambda_noobj': trial.suggest_float(0.1, 1),
    }

    model = build_yolo_model(params)
    history = model.fit(train_data, epochs=params['epochs'], ...)
    val_loss = evaluate_yolo(model, val_data)

    return val_loss
```

---

## 🔚 Summary Table

| Type                | Hyperparameters                           |
| ------------------- | ----------------------------------------- |
| **Architecture**    | Backbone, Anchors, Activation, Input Size |
| **Training**        | Optimizer, LR, Batch Size, Epochs         |
| **Post-processing** | IOU Threshold, Objectness Threshold       |
| **Regularization**  | Dropout, Weight Decay, Augmentation       |
| **Loss Weights**    | λ\_coord, λ\_cls, λ\_noobj                |

---
