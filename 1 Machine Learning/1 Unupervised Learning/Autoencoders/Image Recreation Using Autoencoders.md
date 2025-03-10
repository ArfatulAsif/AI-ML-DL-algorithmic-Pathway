# Recreating Image Using Autoencoders  

## Description  
This project demonstrates the implementation of autoencoders for image reconstruction. Autoencoders are neural networks that aim to compress and then reconstruct input images.  

## Features  
- Image encoding using convolutional layers  
- Image decoding using upsampling layers  
- Reconstruction of images using deep learning techniques  

## Technologies Used  
- Python  
- Keras  
- OpenCV  
- NumPy  
- Matplotlib  

```python
import numpy as np  
from keras.preprocessing.image import img_to_array  # Convert image into an array  
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D # Dense, Conv2D and MaxPooling2D are Neural network layers
                                                                   # UpSampling2D is Decoding layer
from keras.models import Sequential  
import cv2  
import matplotlib.pyplot as plt  # For visualization
```
```python
np.random.seed(42)
```
 
```python
img_size = 256  # Image Size = 256
img_data = []
```

## Converting Image to Array
```python
img = cv2.imread('car.jpg',1)
rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
rgb_img = cv2.resize(rgb_img,(256,256))
img_data.append(img_to_array(rgb_img))
img_final = np.reshape(img_data,(len(img_data),256,256,3))
img_final = img_final.astype('float32')/255
```

## Creating Model
### ANN(Artificial Neural Network) Part
```python
model = Sequential()
model.add(Conv2D(64,(3,3), activation = 'relu' , padding = 'same', input_shape=(256,256,3)))
model.add(MaxPooling2D((2,2), padding = 'same'))
model.add(Conv2D(32,(3,3), activation = 'relu' , padding = 'same'))
model.add(MaxPooling2D((2,2), padding = 'same'))
model.add(Conv2D(16,(3,3), activation = 'relu' , padding = 'same'))
model.add(MaxPooling2D((2,2), padding = 'same'))
```
### Decoding Part
```python
model.add(Conv2D(16,(3,3), activation = 'relu' , padding = 'same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(32,(3,3), activation = 'relu' , padding = 'same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(64,(3,3), activation = 'relu' , padding = 'same'))
model.add(UpSampling2D((2,2)))

model.add(Conv2D(3,(3,3), activation = 'relu' , padding = 'same'))

model.compile(optimizer='adam', loss = 'mean_squared_error', metrics = ['accuracy'])
```

```python
model.summary()
```
## Model: Sequential  

![Model Architecture](Sequence.png)


