#pip install scipy==1.1.0
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
from skimage.filters import threshold_otsu
import numpy as np
from glob import glob
import scipy.misc
from matplotlib.patches import Circle,Ellipse
from matplotlib.patches import Rectangle
import os
from PIL import Image

import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
%matplotlib inline
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization


data = glob('./drive/My Drive/fingerprint/DB*/*')
len(data)

images = []
def read_images(data):
    for i in range(len(data)):
        img = scipy.misc.imread(data[i])
        img = scipy.misc.imresize(img,(224,224))
        images.append(img)
    return images

images = read_images(data)

images_arr = np.asarray(images)
images_arr = images_arr.astype('float32')
images_arr.shape
#(320, 224, 224)
#Data Exploration
print("Dataset (images) shape: {shape}".format(shape=images_arr.shape))

# Display the first image in training data
for i in range(5):
    plt.figure(figsize=[5, 5])
    curr_img = np.reshape(images_arr[i], (224,224))
    plt.imshow(curr_img, cmap='gray')
    plt.show()


#You'll first convert each 224 x 224 image of the dataset into a matrix of size 224 x 224 x 1, which you can then feed into the network:

images_arr = images_arr.reshape(-1, 224,224, 1)
images_arr.shape
 
images_arr.dtype


#rescale the data with the maximum pixel value of the images in the data:
np.max(images_arr)
images_arr = images_arr / np.max(images_arr)

np.max(images_arr), np.min(images_arr)

from sklearn.model_selection import train_test_split
train_X,valid_X,train_ground,valid_ground = train_test_split(images_arr,images_arr,test_size=0.2,random_state=13)


#The Convolutional Autoencoder
"""
The images are of size 224 x 224 x 1 or a 50,176-dimensional vector. You convert the image matrix to an array, rescale it between 0 and 1, reshape it so that it's of size 224 x 224 x 1, and feed this as an input to the network.

Also, you will use a batch size of 128 using a higher batch size of 256 or 512 is also preferable it all depends on the system you train your model. It contributes heavily in determining the learning parameters and affects the prediction accuracy. You will train your network for 50 epochs.
"""
batch_size = 256
epochs = 300
inChannel = 1
x, y = 224, 224
input_img = Input(shape = (x, y, inChannel))

def autoencoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)

    #decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

autoencoder = Model(input_img, autoencoder(input_img))

autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))

#Validation and training loss.
loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(300)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#Prediction
pred = autoencoder.predict(valid_X)


#Reconstruction of Test Images
plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(valid_ground[i, ..., 0], cmap='gray')
plt.show()    
plt.figure(figsize=(20, 4))
print("Reconstruction of Test Images")
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(pred[i, ..., 0], cmap='gray')  
plt.show()


score = model.evaluate([X_test], [y_test], verbose=0)
print("Score: ",score[1]*100)


"""
From the above figures, you can observe that your model did a fantastic job of reconstructing the test images that you predicted using the model. 
At least visually, the test and the reconstructed images look almost exactly similar.
"""

