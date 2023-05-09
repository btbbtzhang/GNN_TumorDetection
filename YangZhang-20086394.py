#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import random
import time
import numpy as np
from math import ceil
import logging


from PIL import Image
from PIL import ImageFilter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, Input, BatchNormalization, UpSampling2D, MaxPooling2D
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import Model, load_model


# In[ ]:


import os
from random import sample
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from ipywidgets import IntProgress
from IPython.display import display, HTML

import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


import numpy as np
ultrasound_size = 128
num_classes = 2
num_epochs = 100
batch_size = 32
max_learning_rate = 0.02
min_learning_rate = 0.00001
regularization_rate = 0.0001
filter_multiplier = 8
class_weights = np.array([0.1, 0.9])
learning_rate_decay = (max_learning_rate - min_learning_rate) / num_epochs


# In[ ]:


pwd


# In[ ]:


local_data_folder = "/global/home/hpc4535/YangSecurity/cisc881/UltrasonicData/"
training_ultrasound_filenames = ["ultrasound-012.npy", "ultrasound-018.npy","ultrasound-024.npy","ultrasound-006-v02.npy"]
training_segmentation_filenames = ["segmentation-012.npy","segmentation-018.npy","segmentation-024.npy","segmentation-006-v02.npy"]

testing_ultrasound_filenames = ["ultrasound-006-v02.npy"]
testing_segmentation_filenames = ["segmentation-006-v02.npy"]


# In[ ]:


# These subfolders will be created/populated in the data folder

data_arrays_folder    = "DataArrays"
notebooks_save_folder = "SavedNotebooks"
results_save_folder   = "SavedResults"
models_save_folder    = "SavedModels"
val_data_folder       = "PredictionsValidation"

data_arrays_fullpath = os.path.join(local_data_folder, data_arrays_folder)
notebooks_save_fullpath = os.path.join(local_data_folder, notebooks_save_folder)
results_save_fullpath = os.path.join(local_data_folder, results_save_folder)
models_save_fullpath = os.path.join(local_data_folder, models_save_folder)
val_data_fullpath = os.path.join(local_data_folder, val_data_folder)

if not os.path.exists(data_arrays_fullpath):
    os.makedirs(data_arrays_fullpath)
    print("Created folder: {}".format(data_arrays_fullpath))

if not os.path.exists(notebooks_save_fullpath):
    os.makedirs(notebooks_save_fullpath)
    print("Created folder: {}".format(notebooks_save_fullpath))

if not os.path.exists(results_save_fullpath):
    os.makedirs(results_save_fullpath)
    print("Created folder: {}".format(results_save_fullpath))

if not os.path.exists(models_save_fullpath):
    os.makedirs(models_save_fullpath)
    print("Created folder: {}".format(models_save_fullpath))

if not os.path.exists(val_data_fullpath):
    os.makedirs(val_data_fullpath)
    print("Created folder: {}".format(val_data_fullpath))


# In[ ]:


import datetime
save_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
print("Save timestamp: {}".format(save_timestamp))


# In[ ]:


# Read all data into numpy arrays
n_files=4
ultrasound_arrays = []
segmentation_arrays = []

f = IntProgress(min=0, max=n_files * 2)
display(f)

time_start = datetime.datetime.now()

for i in range(n_files):
    ultrasound_fullname = os.path.join(data_arrays_fullpath, training_ultrasound_filenames[i])
    segmentation_fullname = os.path.join(data_arrays_fullpath, training_segmentation_filenames[i])

    ultrasound_data = np.load(ultrasound_fullname)
    f.value = i * 2 + 1
    
    segmentation_data = np.load(segmentation_fullname)
    f.value = i * 2 + 2
    
    ultrasound_arrays.append(ultrasound_data)
    segmentation_arrays.append(segmentation_data)



time_stop = datetime.datetime.now()
print("\nTotal time to load from files: {}".format(time_stop - time_start))


# In[ ]:


type(segmentation_arrays)


# In[ ]:


test1=np.load("/global/home/hpc4535/YangSecurity/cisc881/UltrasonicData/training/ultrasound-024.npy")
print(len(test1),len(test1[0]),len(test1[0][0]),len(test1[0][0][0]))


# In[ ]:





# In[ ]:


ultrasound_All_data = np.zeros(
    [0, ultrasound_arrays[0].shape[1], ultrasound_arrays[0].shape[2], ultrasound_arrays[0].shape[3]])
segmentation_All_data = np.zeros(
    [0, ultrasound_arrays[0].shape[1], ultrasound_arrays[0].shape[2], ultrasound_arrays[0].shape[3]])
ultrasound_All_data = np.concatenate(
    (ultrasound_arrays[0], ultrasound_arrays[1], ultrasound_arrays[2], ultrasound_arrays[3]))    
segmentation_All_data = np.concatenate(
    (segmentation_arrays[0], segmentation_arrays[1], segmentation_arrays[2], segmentation_arrays[3]))
filename = training_ultrasound_filenames[0] + " " + training_ultrasound_filenames[1] + " " + training_ultrasound_filenames[2]+ " " + training_ultrasound_filenames[3]
ImageNumbers = ultrasound_All_data.shape[0]
print("\nTotal image number is {} images".format(ImageNumbers))
print("\nImage shape:  {} ".format(ultrasound_All_data.shape))
print("\nSegmentation shape:  {} ".format(segmentation_All_data.shape))


# In[ ]:


print(len(ultrasound_arrays[0]),len(ultrasound_arrays[1]),len(ultrasound_arrays[2]),len(ultrasound_arrays[3]))


# In[ ]:





# In[ ]:


# load data into training and test dataset
# Prepare data arrays
test_ultrasound_data = np.zeros(
    [0, ultrasound_arrays[0].shape[1], ultrasound_arrays[0].shape[2], ultrasound_arrays[0].shape[3]])
test_segmentation_data = np.zeros(
    [0, ultrasound_arrays[0].shape[1], ultrasound_arrays[0].shape[2], ultrasound_arrays[0].shape[3]])

train_ultrasound_data = np.concatenate((ultrasound_arrays[1],ultrasound_arrays[2], ultrasound_arrays[3]))
train_segmentation_data = np.concatenate((segmentation_arrays[1],ultrasound_arrays[2], segmentation_arrays[3]))
train_ultrasound_filename = training_ultrasound_filenames[1] + " " + training_ultrasound_filenames[2] + " " + training_ultrasound_filenames[3]
    
for test_index in range(n_files):
    if test_index != 1 and test_index != 2 and test_index != 3:
        test_ultrasound_data = np.concatenate((test_ultrasound_data, ultrasound_arrays[test_index]))
        test_segmentation_data = np.concatenate((test_segmentation_data, segmentation_arrays[test_index]))
    
n_train = train_ultrasound_data.shape[0]
n_test = test_ultrasound_data.shape[0]
    
print("\n*** Leave-two-out round # {}".format(i))
print("\nTraining on {} images, testidating on {} images...".format(n_train, n_test))
 
test_segmentation_data_onehot = tf.keras.utils.to_categorical(test_segmentation_data, num_classes)

print("testidation ultrasound data shape:            {}".format(test_ultrasound_data.shape))
print("testidation segmentation data shape:          {}".format(test_segmentation_data.shape))
print("testidation segmentation data (onehot) shape: {}".format(test_segmentation_data_onehot.shape))
    
print("Training ultrasound data shape:            {}".format(train_ultrasound_data.shape))
print("Training segmentation data shape:          {}".format(train_segmentation_data.shape))
print("Train_ultrasound_filename: {}".format(train_ultrasound_filename))
    


# In[ ]:





# In[ ]:





# In[ ]:


# define loss function that have been used in backforward
from keras import backend as K
def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


# In[ ]:


## Unet model. Feature size is equal to the image height
feature_n = 128
step_size = 1e-5
input_image_size = 128
###### Question 2 (g) U-net Deep Learning model. Source: https://github.com/SlicerIGT/aigt/blob/master/Notebooks/Segmentation/SegmentationTraining.ipynb
def Gunet(pretrained_weights = None, input_size = (input_image_size, input_image_size, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(feature_n // 16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(feature_n // 16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(feature_n // 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(feature_n // 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(feature_n // 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(feature_n // 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(feature_n // 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(feature_n // 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(feature_n, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(feature_n, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(feature_n // 2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(feature_n // 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(feature_n // 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(feature_n // 4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(feature_n // 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(feature_n // 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(feature_n // 8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(feature_n // 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(feature_n // 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(feature_n // 16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(feature_n // 16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(feature_n // 16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = step_size), loss =dice_coef_loss, metrics = ['accuracy'])
    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
#"binary_crossentropy" dice_coef_loss
    return model


# In[ ]:


import numpy as np
import os 

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

model = unet()

model.summary()


# In[ ]:





# In[ ]:


### graph layer
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import scipy
import scipy.signal
import datetime
from scipy.spatial.distance import cdist

# Graph data converting based on fixed Gabor filter
# 4 dimensional image pack, includes (slice, width, heigth, channel). Inputs are normalized already
def SimpleGraphMatrixCalulation(Img_pack): 
    time_start = datetime.datetime.now()
    GraphImagePack=[]
    for i in range(len(Img_pack)):
        print("\n*** Start converting to graph data------Done image: ", i , ".Left images: ", len(Img_pack)-i)
        img = Img_pack[i]
        # create graph data based on nodes (graph data value is decided by the position of pairwise nodes)
        N = img.shape[0]
        col, row = np.meshgrid(np.arange(N), np.arange(N))
        coord = np.stack((col, row), axis=2).reshape(-1, 2) / N
        dist = cdist(coord, coord)  
        sigma = 0.2 * np.pi  # width of a Gaussian
        A = np.exp(- dist / sigma ** 2)  
        
        
        #x, y = np.meshgrid(np.arange(-float(N), N), np.arange(-float(N), N))
        #y = skimage.transform.rotate(y, 35)
        ##x2 = skimage.transform.rotate(x, -35)
        #sigma = 0.75 * np.pi
        #lmbda = 1.5 * sigma
        #gamma = 1.3
        #gabor = np.exp(-(x**2 + gamma*y**2)/(2*sigma**2))*np.cos(2*np.pi*x2/lmbda)
        # Create the adjacency matrix based on the Gabor filter 
        #A = np.zeros((N ** 2, N ** 2))
        #for j in range(N):
        #    for k in range(N):
        #        A[j*N + k, :] = gabor[N - j:N - j + N, N - k:N - k + N].flatten()
        # add the affinity to the original data
        Img_graph = A.dot(img.reshape(-1, 1)).reshape(N, N)
        print(Img_graph.shape)
        GraphImagePack.append(Img_graph)
    time_stop = datetime.datetime.now()
    print("\nTotal time to convert from files: {}".format(time_stop - time_start))
    return GraphImagePack


# In[ ]:





# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import scipy
import scipy.signal
import datetime
from scipy.spatial.distance import cdist

'''
Graph Convolution Layer according to (T. Kipf and M. Welling, ICLR 2017)
Additional tricks (power of adjacency matrix and weight self connections) as in the Graph U-Net paper
Paper Ref: https://arxiv.org/pdf/1609.02907.pdf
'''

# 4 dimensional image pack, includes (slice, width, heigth, channel). Inputs are normalized already
def GraphMatrixCalulation(Img_pack): 
    time_start = datetime.datetime.now()
    GraphImagePack=[]
    #loadImgSize = len(Img_pack)
    loadImgSize = 2 # test two images
    for i in range(loadImgSize):
        print("\n*** Start converting to graph data------Done image: ", i , ".Left images: ", len(Img_pack)-i)
        img = Img_pack[i]
        # create graph data based on nodes (graph data value is decided by the position of pairwise nodes)
        N = img.shape[0] # image size
        x, y = np.meshgrid(np.arange(-float(N), N), np.arange(-float(N), N))
        coord = np.stack((x, y), axis=2).reshape(-1, 2) / N
        dist = cdist(coord, coord)
        sigma = 0.05 * np.pi
        A = np.exp(- dist / sigma ** 2)
        A[A < 0.01] = 0
        
        # Normalization as per (Kipf & Welling, ICLR 2017)
        D = A.sum(1)  # nodes degree (N,)
        
        I = np.identity(A.shape[0])
        I = np.float8(I)
        A_hat = np.float8(A+I)
        D_hat = (D + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat  # N,N

        # Some additional trick I found to be useful
        L[L > 0.0001] = L[L > 0.0001] - 0.2
        
        Img_graph = L.dot(img.reshape(-1, 1)).reshape(N, N)
        GraphImagePack.append(Img_graph)
    time_stop = datetime.datetime.now()
    print("\nTotal time to convert from files: {}".format(time_stop - time_start))
    return GraphImagePack


# In[ ]:


## convert data into graph data by using a simple way
GraphUltrasonic_data_train = SimpleGraphMatrixCalulation(train_ultrasound_data)


# In[ ]:


type(GraphUltrasonic_data_train)


# In[ ]:


GraphUltrasonic_data1 = SimpleGraphMatrixCalulation(testData)


# In[ ]:


GraphUltrasonic_data_All = GraphMatrixCalulation(ultrasound_All_data)


# In[ ]:


# resize to the required input size for the model, which is (128, 128, 1)
for i in range(len(GraphUltrasonic_data_All)):
    GraphUltrasonic_data_All[i].shape = (128,128,1)
GraphUltrasonic_data_All = np.array(GraphUltrasonic_data_All)


# In[ ]:


type(GraphUltrasonic_data_All)
GraphUltrasonic_data_All.shape


# In[ ]:





# In[ ]:


# save the all graph conveted data
from numpy import asarray
from numpy import save
save('GraphUltrasonic_data_All_new.npy', GraphUltrasonic_data_All)


# In[ ]:


## load the previous calculated graph data
GraphUltrasonic_data_All=np.load('GraphUltrasonic_data_All.npy')


# In[ ]:


GraphUltrasonic_data_All.shape


# In[ ]:


ultrasound_All_data.shape


# In[ ]:


segmentation_All_data.shape


# In[ ]:


testGraphData =GraphUltrasonic_data_All[940:,:,:,:]
testGraphData.shape


# In[ ]:


testData = ultrasound_All_data[940:,:,:,:]
testData.shape


# In[ ]:


trainData = ultrasound_All_data[:940,:,:,:]
trainData.shape


# In[ ]:


testDataSeg = segmentation_All_data[940:,:,:,:]
testDataSeg.shape


# In[ ]:


trainGraphData = GraphUltrasonic_data_All[:940,:,:,:]
trainGraphData.shape


# In[ ]:


trainDataSeg = segmentation_All_data[:940,:,:,:]
trainDataSeg.shape


# In[ ]:


512/128


# In[ ]:


pwd


# In[ ]:





# In[ ]:


train_segmentation_data.shape


# In[ ]:


epoch_n = 100
batch_size = 30

model.fit(trainData, trainDataSeg, batch_size=batch_size, epochs=epoch_n, verbose=1,validation_split=0.2, shuffle=False)


# In[ ]:


epoch_n = 100
batch_size = 30


model.fit(trainGraphData, trainDataSeg, batch_size=batch_size, epochs=epoch_n, verbose=1,validation_split=0.2, shuffle=True)


# In[ ]:


from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
model.save("model_all_GUnet.h5")
print("Saved model to disk")


# In[ ]:


from numpy import loadtxt
from keras.models import load_model

# load model
model = load_model('model_all_GUnet.h5', custom_objects={'dice_coef_loss': dice_coef_loss})
# summarize model.
model.summary()


# In[ ]:





# In[ ]:


test_ultrasound_data.shape


# In[ ]:


a = pred_segmentation_data


# In[ ]:


pred_segmentation_data[0]


# In[ ]:


a=1000*a


# In[ ]:


a[0]


# In[ ]:


max(test_segmentation_data[18][80])


# In[ ]:


trainGraphData


# np.arange(0, num_test_ultrasound, int(num_test_ultrasound / (num_samples - 1)) - 1)

# In[ ]:


import matplotlib.pyplot as plt

num_test_ultrasound = testData.shape[0]
num_test_segmentation = trainDataSeg.shape[0]

pred_segmentation_data = model.predict(testGraphData)

num_samples = 5
#image_indices = np.arange(0, num_test_ultrasound, int(num_test_ultrasound / (num_samples - 1)) - 1)
image_indices = [0,5,8,13,10]

f = plt.figure(figsize=(18,9))

for i in range(num_samples):
    subplot = f.add_subplot(3, 5, i + 1)
    subplot.axis("Off")
    image_index = image_indices[i]
    plt.imshow(pred_segmentation_data[image_index, :, :, 0].astype(np.float32))
    plt.title("Pred " + str(image_index))

for i in range(num_samples):
    subplot = f.add_subplot(3, 5, 5 + i + 1)
    subplot.axis("Off")
    image_index = image_indices[i]
    plt.imshow(test_segmentation_data[image_index, :, :, 0].astype(np.float32))
    plt.title("Segm " + str(image_index))

for i in range(num_samples):
    subplot = f.add_subplot(3, 5, 10 + i + 1)
    subplot.axis("Off")
    image_index = image_indices[i]
    plt.imshow(test_ultrasound_data[image_index, :, :, 0].astype(np.float32))
    plt.title("Image " + str(image_index))
##### 10 epochs
plt.show()


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt

num_test_ultrasound = testData.shape[0]
num_test_segmentation = trainDataSeg.shape[0]

pred_segmentation_data = model.predict(testGraphData)

num_samples = 5
#image_indices = np.arange(0, num_test_ultrasound, int(num_test_ultrasound / (num_samples - 1)) - 1)
image_indices = [0,5,10,15,20]

f = plt.figure(figsize=(18,9))

for i in range(num_samples):
    subplot = f.add_subplot(3, 5, i + 1)
    subplot.axis("Off")
    image_index = image_indices[i]
    plt.imshow(pred_segmentation_data[image_index, :, :, 0].astype(np.float32))
    plt.title("Pred " + str(image_index))

for i in range(num_samples):
    subplot = f.add_subplot(3, 5, 5 + i + 1)
    subplot.axis("Off")
    image_index = image_indices[i]
    plt.imshow(test_segmentation_data[image_index, :, :, 0].astype(np.float32))
    plt.title("Segm " + str(image_index))

for i in range(num_samples):
    subplot = f.add_subplot(3, 5, 10 + i + 1)
    subplot.axis("Off")
    image_index = image_indices[i]
    plt.imshow(test_ultrasound_data[image_index, :, :, 0].astype(np.float32))
    plt.title("Image " + str(image_index))
##### 28 epochs
plt.show()


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt

num_test_ultrasound = testData.shape[0]
num_test_segmentation = trainDataSeg.shape[0]

pred_segmentation_data = model.predict(testGraphData)

num_samples = 5
#image_indices = np.arange(0, num_test_ultrasound, int(num_test_ultrasound / (num_samples - 1)) - 1)
image_indices = [0,5,10,15,20]

f = plt.figure(figsize=(18,9))

for i in range(num_samples):
    subplot = f.add_subplot(3, 5, i + 1)
    subplot.axis("Off")
    image_index = image_indices[i]
    plt.imshow(pred_segmentation_data[image_index, :, :, 0].astype(np.float32))
    plt.title("Pred " + str(image_index))

for i in range(num_samples):
    subplot = f.add_subplot(3, 5, 5 + i + 1)
    subplot.axis("Off")
    image_index = image_indices[i]
    plt.imshow(test_segmentation_data[image_index, :, :, 0].astype(np.float32))
    plt.title("Segm " + str(image_index))

for i in range(num_samples):
    subplot = f.add_subplot(3, 5, 10 + i + 1)
    subplot.axis("Off")
    image_index = image_indices[i]
    plt.imshow(test_ultrasound_data[image_index, :, :, 0].astype(np.float32))
    plt.title("Image " + str(image_index))
##### 49 epochs
plt.show()


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt

num_test_ultrasound = testData.shape[0]
num_test_segmentation = trainDataSeg.shape[0]

pred_segmentation_data = model.predict(testData)

num_samples = 6
#image_indices = np.arange(0, num_test_ultrasound, int(num_test_ultrasound / (num_samples - 1)) - 1)
image_indices = [0,18,19,20,3,1]

f = plt.figure(figsize=(18,9))

for i in range(num_samples):
    subplot = f.add_subplot(3, 6, i + 1)
    subplot.axis("Off")
    image_index = image_indices[i]
    plt.imshow(pred_segmentation_data[image_index, :, :, 0].astype(np.float32))
    plt.title("Pred " + str(image_index))

for i in range(num_samples):
    subplot = f.add_subplot(3, 6, 6 + i + 1)
    subplot.axis("Off")
    image_index = image_indices[i]
    plt.imshow(test_segmentation_data[image_index, :, :, 0].astype(np.float32))
    plt.title("Segm " + str(image_index))

for i in range(num_samples):
    subplot = f.add_subplot(3, 6, 12 + i + 1)
    subplot.axis("Off")
    image_index = image_indices[i]
    plt.imshow(test_ultrasound_data[image_index, :, :, 0].astype(np.float32))
    plt.title("Image " + str(image_index))
##### 60
plt.show()


# In[ ]:


testData =GraphUltrasonic_data_All[940:,:,:,:]
testData.shape


# In[ ]:


import matplotlib.pyplot as plt

num_test_ultrasound = testData.shape[0]
num_test_segmentation = trainDataSeg.shape[0]

pred_segmentation_data = model.predict(testData)

num_samples = 6
#image_indices = np.arange(0, num_test_ultrasound, int(num_test_ultrasound / (num_samples - 1)) - 1)
image_indices = [0,18,19,20,3,1]

f = plt.figure(figsize=(18,9))

for i in range(num_samples):
    subplot = f.add_subplot(3, 6, i + 1)
    subplot.axis("Off")
    image_index = image_indices[i]
    plt.imshow(pred_segmentation_data[image_index, :, :, 0].astype(np.float32))
    plt.title("Pred " + str(image_index))

for i in range(num_samples):
    subplot = f.add_subplot(3, 6, 6 + i + 1)
    subplot.axis("Off")
    image_index = image_indices[i]
    plt.imshow(test_segmentation_data[image_index, :, :, 0].astype(np.float32))
    plt.title("Segm " + str(image_index))

for i in range(num_samples):
    subplot = f.add_subplot(3, 6, 12 + i + 1)
    subplot.axis("Off")
    image_index = image_indices[i]
    plt.imshow(test_ultrasound_data[image_index, :, :, 0].astype(np.float32))
    plt.title("Image " + str(image_index))
##### 100
plt.show()


# In[ ]:


pred_segmentation_data.shape


# In[ ]:


a = np.squeeze(pred_segmentation_data[2])


# In[ ]:


a.shape


# In[ ]:





# In[ ]:




