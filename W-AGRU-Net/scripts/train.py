# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 02:39:58 2024

@author: win_10
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 03:05:59 2024

@author: win_10
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 00:22:42 2024

@author: win_10
"""

#Implementation
#Import Packages & Custom Libraries------------(1)
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf

from wagru_net import Dual_agruNet
from generator import ImageDataGen
from downloader import ImageDataExtractor
#from ipywidgets import FloatProgress
import ipywidgets 
from tensorflow.keras.models import load_model

#Download & Extract Data---------------(2)
dataExtractor = ImageDataExtractor(RemoveTemporaryFiles = True)
MAT_DATA_PATH, IMG_DATA_PATH, MASK_DATA_PATH, DATA_README_PATH = dataExtractor.downloadAndExtractImages()

#Verify the Data Paths----------------------(3)
MAT_DATA_PATH, IMG_DATA_PATH, MASK_DATA_PATH, DATA_README_PATH

#Define Hyperparameters-------------------(4)
image_size = 128
image_channels = 1

epochs = 100
batch_size = 2

# there are a total of 3064 images.
# so fixing 2900 of data available for training set
# 200 for validation set and 64 for test set.
validation_data_size = 450
test_data_size = 164
train_data_size = 2450

#Define Reusable Functions---------------------(5)
def VisualizeImageAndMask(image, mask, prediction_img = None):
    
    """
    
    Displays the image, mask and the predicted mask
    of the input image.
    
    Args:
        image: the original image.
        mask: the given mask of the image.
        prediction_img: the predicted mask of the image.
        
    Return:
        None
        
    """
    fig = plt.figure()
    fig.subplots_adjust(hspace = 0.6, wspace = 0.6)
    fig.suptitle('Image & Mask(s)', fontsize = 15)
    fig.subplots_adjust(top = 1.15)
    
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(image, cmap = "gray")
    setTitleAndRemoveTicks(ax, 'Original\nImage')
    
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(mask, cmap = "gray")
    setTitleAndRemoveTicks(ax, 'Original\nMask')
    
    if prediction_img is not None:
        #prediction_img = prediction_img * 255
        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(np.reshape(prediction_img, (image_size, image_size)), cmap = "gray")
        setTitleAndRemoveTicks(ax, 'Predicted\nMask')
    
def setTitleAndRemoveTicks(axes, title):
    
    """
    Sets the sub-plot title and removes the 
    x & y ticks on the respective axes.
    
    Args:
        axes: the subplot.
        title: title of the subplot.
        
    Return:
        None
        
    """
    
    # set plot title
    axes.title.set_text(title)
    
    # remove the ticks
    axes.set_xticks([])
    axes.set_yticks([])

#Prepare Data for Segmentation Task----------------(6)
# get the ids of the images.
# os.walk yields a 3-tuple (dirpath, dirnames, filenames). We need the directory names here.
# IMG_DATA_PATH = 'data\\imgData\\img'
# MASK_DATA_PATH = 'data\\imgData\\mask'
image_ids = next(os.walk(IMG_DATA_PATH))[2]
np.random.shuffle(image_ids)

# partition the data into train, test and validation sets.
testing_data_ids = image_ids[:test_data_size]
validation_data_ids = image_ids[:validation_data_size]
training_data_ids = image_ids[:train_data_size]

#Image Data Generator - Verification----------------(7)
temp_data_generator = ImageDataGen(image_ids = training_data_ids,
                                   img_path = IMG_DATA_PATH, 
                                   mask_path = MASK_DATA_PATH,
                                   batch_size = batch_size, 
                                   image_size = image_size)

# get one batch of data
images, masks = temp_data_generator.__getitem__(0)
print("Batch Dimension Details:", images.shape, masks.shape)

VisualizeImageAndMask(image = images[1], mask = masks[1])
temp_data_generator = None

#Model Training and Validation------------------)8()
#Model Initialization & Compilation

# Initialize the Unet++ with the default parameters. 
# The default params are the one that were used in the original paper.
# Input shape - (512, 512, 1), 
# filters [32, 64, 128, 256, 512].
conn_atten_res = Dual_agruNet(input_shape = (128, 128, 1), deep_supervision = False)

# call the build netowrk API to build the network.
model = conn_atten_res.get_arwnet()
# compile & summarize the model
if model is not None:
    conn_atten_res.CompileAndSummarizeModel(model = model)

#Initialize the Data Generators----------------------(9)
train_gen = ImageDataGen(image_ids = training_data_ids,
                         img_path = IMG_DATA_PATH, 
                         mask_path = MASK_DATA_PATH, 
                         image_size = image_size, 
                         batch_size = batch_size)

valid_gen = ImageDataGen(image_ids = validation_data_ids, 
                         img_path = IMG_DATA_PATH, 
                         mask_path = MASK_DATA_PATH,
                         image_size = image_size, 
                         batch_size = batch_size)

test_gen = ImageDataGen(image_ids = testing_data_ids, 
                        img_path = IMG_DATA_PATH, 
                        mask_path = MASK_DATA_PATH,
                        image_size = image_size, 
                        batch_size = batch_size)

train_steps = len(training_data_ids)//batch_size
valid_steps = len(validation_data_ids)//batch_size


train_steps, valid_steps
###################################Model Training#######################################
from keras.callbacks import CSVLogger
csv_logger = CSVLogger('C:/Users/win_10/conn_atten_res2.log')
start=time.time()

history =model.fit(train_gen, 
          validation_data = valid_gen, 
          steps_per_epoch = train_steps, 
          validation_steps = valid_steps, callbacks= [csv_logger], 
          epochs = epochs)
end=time.time()
print('elapsed time =',end-start)
model.save("C:/Users/win_10/conn_atten_res2.h5")
#model.save("C:/Users/win_10/try.h5")
print(history.history.keys())

############################################################################
import tensorflow as tf
import pandas as pd
import keras

#history=model = tf.keras.models.load_model("C:/Users/win_10/UNetpp_BrainTumorSegment.h5")
tf.keras.models.load_model("C:/Users/win_10/conn_atten_res2.h5")
# list all data in history
hist = pd.read_csv('C:/Users/win_10/conn_atten_res2.log', sep=',', engine='python')
acc=hist['acc']
val_acc=hist['val_acc']
val_acc

#################محاولة مني########################
# summarize history for accuracy
plt.plot(acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()

# summarize history for dice_coef
plt.plot(history.history['__dice_coef'])
plt.plot(history.history['val___dice_coef'])
plt.title('model Dice_coef')
plt.ylabel('dice_coef')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()


# summarize history for I-OU
plt.plot(history.history['__iou_loss_core'])
plt.plot(history.history['val___iou_loss_core'])
plt.title('model iou_loss_core')
plt.ylabel('iou_loss_core')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()

########################### Model Testing and Prediction Visualizations#############################

test_gen = ImageDataGen(image_ids = testing_data_ids, 
                        img_path = IMG_DATA_PATH, 
                        mask_path = MASK_DATA_PATH,
                        image_size = image_size, 
                        batch_size = 16)

# get the test set images
test_images, test_masks = test_gen.__getitem__(0)
predicted_masks = model.predict(test_images)

predicted_masks = predicted_masks > 0.5

test_images_2, test_masks_2 = test_gen.__getitem__(3)
predicted_masks_2 = model.predict(test_images)

predicted_masks_ = predicted_masks_2 > 0.5

test_images_3, test_masks_3 = test_gen.__getitem__(2)
predicted_masks_3 = model.predict(test_images)

predicted_masks_3= predicted_masks_2 > 0.5

####Viz 1

VisualizeImageAndMask(image = test_images[9], mask = test_masks[9], prediction_img = predicted_masks_[9])
####Viz 2
VisualizeImageAndMask(image = test_images[1], mask = test_masks[1], prediction_img = predicted_masks[1])
VisualizeImageAndMask(image = test_images[2], mask = test_masks[2], prediction_img = predicted_masks[2])

VisualizeImageAndMask(image = test_images[12], mask = test_masks[12], prediction_img = predicted_masks[12])
VisualizeImageAndMask(image = test_images[5], mask = test_masks[5], prediction_img = predicted_masks[5])
VisualizeImageAndMask(image = test_images[31], mask = test_masks[31], prediction_img = predicted_masks[31])
VisualizeImageAndMask(image = test_images[13], mask = test_masks[13], prediction_img = predicted_masks[13])
VisualizeImageAndMask(image = test_images[23], mask = test_masks[23], prediction_img = predicted_masks[23])
VisualizeImageAndMask(image = test_images[25], mask = test_masks[25], prediction_img = predicted_masks[25])
VisualizeImageAndMask(image = test_images[0], mask = test_masks[0], prediction_img = predicted_masks[0])
VisualizeImageAndMask(image = test_images[22], mask = test_masks[22], prediction_img = predicted_masks[22])
VisualizeImageAndMask(image = test_images[8], mask = test_masks[8], prediction_img = predicted_masks[8])
VisualizeImageAndMask(image = test_images[3], mask = test_masks[3], prediction_img = predicted_masks[3])

VisualizeImageAndMask(image = test_images_3[30], mask = test_masks_3[30], prediction_img = predicted_masks_3[30])
VisualizeImageAndMask(image = test_images_3[3], mask = test_masks_3[3], prediction_img = predicted_masks_3[3])
VisualizeImageAndMask(image = test_images_3[10], mask = test_masks_3[10], prediction_img = predicted_masks_3[10])
VisualizeImageAndMask(image = test_images_3[1], mask = test_masks_3[1], prediction_img = predicted_masks_3[1])
VisualizeImageAndMask(image = test_images_3[15], mask = test_masks_3[15], prediction_img = predicted_masks_3[15])
VisualizeImageAndMask(image = test_images_3[25], mask = test_masks_3[25], prediction_img = predicted_masks_3[25])
VisualizeImageAndMask(image = test_images_3[19], mask = test_masks_3[19], prediction_img = predicted_masks_3[19])



VisualizeImageAndMask(image = test_images[9], mask = test_masks[9], prediction_img = predicted_masks[9])


################Model evaluation################################
print("Evaluate on test data")
results = model.evaluate(test_gen, batch_size=8)
