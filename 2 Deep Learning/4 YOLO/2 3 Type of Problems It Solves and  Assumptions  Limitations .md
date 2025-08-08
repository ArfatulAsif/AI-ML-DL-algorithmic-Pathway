Perfect! Based on the structure and tone of your **FNN notes**, here’s your detailed YOLO version of:

---

# 🧠 **2. Type of Problems It Solves**

* [x] **Object Detection**
* [x] **Multi-class Classification**
* [x] **Localization**
* [x] **Tracking (with extensions)**
* [ ] Clustering
* [ ] Dimensionality Reduction
* [ ] Reinforcement Learning

---

### **What type of data is it best for?**

YOLO is designed for **image and video data**, where the goal is to:

* Detect **what** objects are present,
* Determine **where** they are in the frame,
* And do this in **real-time**, even for multiple objects at once.

YOLO works best when:

* Objects are relatively **distinct and well-spaced**,
* You need **speed** and **accuracy** balanced,
* You have **annotated bounding box data** for training.

---

# 🚧 **3. Assumptions & Limitations**

### **Does it assume:**

* [x] **Each object’s center lies in exactly one grid cell**
* [x] **Fixed number of objects per grid cell**
* [x] **Objects can be approximated as rectangles**
* [ ] Independence of features
* [ ] Normal distribution
* [ ] Linearity

> *(YOLO assumes a very specific spatial structure in the image through a grid system, and every detection must "belong" to a grid cell.)*

---

### **Is it sensitive to:**

* [x] **Small Objects**: Can struggle with detecting small or densely packed objects, especially if they fall into the same grid cell.
* [x] **Overlapping Objects**: YOLO may miss objects of the same class that are too close together.
* [x] **Aspect Ratios / Distortions**: YOLO resizes input to a fixed size, which may distort object proportions.
* [ ] Multicollinearity — *(not applicable for image data)*
* [ ] Outliers — *(outliers are less of a concern in pixel space)*

---

## **Preprocessing needed?**

✅ **Yes, definitely.**
You need to:

* Resize all input images to a fixed resolution (e.g., 416×416 or 640×640),
* Normalize pixel values (e.g., scale from 0 to 1),
* Optionally use data augmentation (e.g., flip, rotate, scale, color jitter),
* Format labels in **YOLO format**:

  ```plaintext
  <class_id> <x_center> <y_center> <width> <height>
  ```

> 📦 Boxes should be in **normalized coordinates** \[0, 1], relative to image size.

---