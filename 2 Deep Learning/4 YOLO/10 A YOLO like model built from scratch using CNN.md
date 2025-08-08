
# YOLO Model Implementation (Using Oxford-IIIT Pet Dataset)


**We use a small CNN backbone**



## **1. Data Preprocessing**

We'll load the Oxford-IIIT Pet Dataset, extract images and bounding boxes, and prepare them for YOLO-style grid output.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tensorflow.keras.utils import get_file

# Download the dataset
dataset_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
annotation_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

images_path = get_file("images.tar.gz", dataset_url, extract=True)
anns_path = get_file("annotations.tar.gz", annotation_url, extract=True)

import glob
image_files = glob.glob(images_path.replace('.tar.gz','') + "/*.jpg")
annot_files = glob.glob(anns_path.replace('.tar.gz','') + "/xmls/*.xml")

# Limit to 500 samples for speed
image_files = image_files[:500]
annot_files = annot_files[:500]

# Function to parse XML and extract bounding boxes and labels
def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    boxes = []
    labels = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(label)
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text) / width
        y1 = float(bbox.find('ymin').text) / height
        x2 = float(bbox.find('xmax').text) / width
        y2 = float(bbox.find('ymax').text) / height
        boxes.append([y1, x1, y2, x2])
    return np.array(boxes), labels

# Prepare tf.data.Dataset
def load_and_prepare(img_path, ann_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (128, 128)) / 255.0
    boxes, _ = tf.py_function(parse_annotation, [ann_path], [tf.float32, tf.string])
    return img, boxes

dataset = tf.data.Dataset.from_tensor_slices((image_files, annot_files))
dataset = dataset.map(load_and_prepare).batch(8).prefetch(tf.data.AUTOTUNE)

train_ds = dataset.take(40)
test_ds = dataset.skip(40)
```

**Explanation:**

* We download and extract the dataset images and annotations.
* We parse Pascal VOC XML files for bounding boxes and normalized coordinates.
* We create a `tf.data.Dataset` that loads, resizes, and batches data for training and testing.

---

## **2. Build the YOLO-like Model**

We'll construct a simplified Tiny YOLO-style model with grid output.

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Reshape

def build_tiny_yolo(input_size=128, grid_size=8, num_classes=1):
    inputs = Input(shape=(input_size, input_size, 3))
    x = inputs
    for filters in [16, 32, 64, 128]:
        x = Conv2D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D()(x)
    pred = Conv2D(grid_size * (5 + num_classes), 1, padding='same')(x)
    pred = Reshape((grid_size, grid_size, 5 + num_classes))(pred)
    return Model(inputs, pred)

model = build_tiny_yolo(num_classes=1)
model.summary()
```

**Explanation:**

* We use a small CNN backbone followed by a final conv layer that predicts for each grid cell: bounding box (x, y, w, h), objectness, and one class probability.

---

## **3. Loss Function & Training Behavior**

We define a simplified YOLO loss: MSE for coordinates, BCE for objectness, and CE for class probability.

```python
import tensorflow.keras.backend as K

def yolo_loss(y_true, y_pred):
    obj_mask = y_true[..., 4:5]
    coord_loss = K.sum(obj_mask * K.square(y_true[..., :4] - y_pred[..., :4]))
    conf_loss = K.sum(K.binary_crossentropy(obj_mask, y_pred[..., 4:5]))
    cls_loss = K.sum(obj_mask * K.binary_crossentropy(y_true[..., 5:], y_pred[..., 5:]))
    return coord_loss + conf_loss + cls_loss

model.compile(optimizer='adam', loss=yolo_loss)
```

---

## **4. Training the Model**

We format targets into a grid and train for a few epochs.

```python
def format_target(images, boxes):
    batch_size = images.shape[0]
    S = 8
    target = np.zeros((batch_size, S, S, 6), dtype='float32')
    for i in range(batch_size):
        for b in boxes[i].numpy():
            y1,x1,y2,x2 = b
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            cx = int(x * S)
            cy = int(y * S)
            target[i, cy, cx, :4] = [x, y, w, h]
            target[i, cy, cx, 4] = 1
            target[i, cy, cx, 5] = 1  # single class
    return target

epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for imgs, bxs in train_ds:
        y_true = format_target(imgs, bxs)
        loss = model.train_on_batch(imgs, y_true)
    print("Loss:", loss)
```

---

## **5. Model Evaluation: Visualization**

We visualize predictions on test images with bounding boxes.

```python
def plot_predictions(model, ds):
    S = 8
    for imgs, bxs in ds.take(1):
        y_pred = model.predict(imgs)
        for i in range(len(imgs)):
            plt.imshow(imgs[i])
            pred = y_pred[i]
            for cy in range(S):
                for cx in range(S):
                    conf = pred[cy, cx, 4]
                    if conf > 0.5:
                        x, y, w, h = pred[cy, cx, :4]
                        x1 = (x - w/2) * 128
                        y1 = (y - h/2) * 128
                        rect = plt.Rectangle((x1, y1), w*128, h*128,
                                             fill=False, color='red')
                        plt.gca().add_patch(rect)
            plt.axis('off')
            plt.show()

plot_predictions(model, test_ds)
```

---

## **6. Summary**

**Data Preprocessing**: Load real dataset (Oxford-IIIT Pet), resize, and prepare bounding boxes.
**Model Building**: Tiny CNN outputs grid predictions for YOLO.
**Loss & Training**: Custom YOLO-like loss applied; model trained for simple data.
**Evaluation**: Visualize results with bounding box overlays.

---

This matches your FNN template using **real dataset**, end-to-end from loading annotations to training and visualization. For a production-ready YOLO (v3/v5), you'd incorporate anchor boxes, multi-scale heads, NMS, and mAP evaluation. Let me know if you want to extend this!
