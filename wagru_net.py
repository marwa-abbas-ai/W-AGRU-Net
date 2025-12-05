# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 08:48:40 2024

@author: win_10
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 04:50:32 2024

@author: win_10
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 01:31:54 2024

@author: win_10
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, concatenate, Conv2DTranspose, Add, UpSampling2D
from tensorflow.keras.layers import MaxPooling2D, Dropout, Activation, BatchNormalization,Flatten, Reshape, GlobalAveragePooling2D, Multiply, Lambda
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.layers import InputLayer, Activation, add, multiply


from tensorflow.keras.losses import binary_crossentropy




class Dual_agruNet():
    
    """ 
    Unet++ Model design.
    
    This module consumes the Unet utilities framework moule and designs the Unet network.
    It consists of a contracting path and an expansive path. Both these paths are joined 
    by a bottleneck block.
    
    The different blocks involved in the design of the network can be referenced @ 
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    
    Reference:
        [1] UNet++: A Nested U-Net Architecture for Medical Image Segmentation.
            https://arxiv.org/abs/1807.10165
            
        [2] https://paperswithcode.com/paper/unet-a-nested-u-net-architecture-for-medical
        
    """
  
   
    def __init__(self, input_shape = (512, 512, 1), filters = [32, 64, 128, 256, 512], nb_classes =1, deep_supervision = False):
        
        """
        Initialize the Unet framework and the model parameters - input_shape, 
        filters and padding type. 
        
        Args:
            input_shape (tuple): A shape tuple (integers), not including the batch size.
                                 Default value is (512, 512, 1).
                                 
            filters (array of integers: a collection of filters denoting the number of components to be used at each blocks along the 
                        contracting and expansive paths. The original paper implementation for number of filters along the 
                        contracting and expansive paths are [32, 64, 128, 256, 512]. (as per paper: k = 32 × 2^i).
                        
            nb_classes (Integer): the dimensionality (no. of filters) of the output space .
                        (i.e. the number of output filters in the convolution).

            deep_supervision (boolean): A flag that toggles between the 2 different training modes -
                                        1) the ACCURATE mode - where the outputs from all segmentation 
                                           branches are averaged., 
                                        2) the FAST mode - wherein the final segmentation map is selected from 
                                           only one of the segmentation branches.
                                        Default vaue - False
            
        **Remarks: The default values are as per the implementation in the original paper @ https://arxiv.org/pdf/1505.04597
         
        """

        self.__input_shape = input_shape
        self.__filters = filters
        self.__nb_classes = nb_classes
        self.__deep_supervision = deep_supervision
        self.__smooth = 1. # Used to prevent the denominator from 0 when computing the DICE coefficient.
        
   
    

    def get_arwnet(self):
        input_img = Input(shape = self.__input_shape, name = 'InputLayer')
        conv1 = self.__resnet_block(input_img,32 , strides=1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = self.__resnet_block(pool1,64 , strides=1)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.__resnet_block(pool2, 128, strides=1)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.__resnet_block(pool3, 256, strides=1)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
        conv5 = self.aspp_block(pool4, 512)
        conv5 = self.__resnet_block(pool4, 512, strides=1)

     
        gating = self.UnetGatingSignal(conv5)
        attn_1 = self.AttnGatingBlock(conv4, gating, 256)
        up6 = concatenate([Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv5), attn_1], axis=3)  
    
        conv6 = self.__resnet_block(up6, 256, strides=1)

    
        gating = self.UnetGatingSignal(conv6)
        attn_2 = self.AttnGatingBlock(conv3, gating, 128)
        up7 = concatenate([Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv6), attn_2], axis=3) 
    
        conv7 = self.__resnet_block(up7, 128, strides=1)

        gating = self.UnetGatingSignal(conv7)
        attn_3 = self.AttnGatingBlock(conv2, gating, 64)
        up8 = concatenate([Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv7), attn_3], axis=3) 
        
        conv8 = self.__resnet_block(up8, 64, strides=1)
   
        
        gating = self.UnetGatingSignal(conv8)
        attn_4 = self.AttnGatingBlock(conv1, gating, 32)
        up9 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv8), attn_4], axis=3) 
        
        conv9 = self.__resnet_block(up9, 32, strides=1)

        down10 = concatenate([Conv2D(32, (3, 3), activation='relu', padding='same')(conv9), conv9], axis=3)
        conv10 = self.__resnet_block(down10, 32, strides=1)  

        pool10 = MaxPooling2D(pool_size=(2, 2))(conv10)
        
        down11 = concatenate([Conv2D(64, (3, 3), activation='relu', padding='same')(pool10), conv8], axis=3)
        conv11 = self.__resnet_block(down11, 64, strides=1)
        pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
        
        down12 = concatenate([Conv2D(128, (3, 3), activation='relu', padding='same')(pool11), conv7], axis=3)
        conv12 = self.__resnet_block(down12, 128, strides=1)

        pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)

        down13 = concatenate([Conv2D(256, (3, 3), activation='relu', padding='same')(pool12), conv6], axis=3)
        conv13 = self.__resnet_block(down13, 256, strides=1)

        pool13 = MaxPooling2D(pool_size=(2, 2))(conv13)
        
        conv14 = self.aspp_block(pool13, 512)
           
        gating = self.UnetGatingSignal(conv14)
        attn_1 = self.AttnGatingBlock(conv13, gating, 256)
        up15 = concatenate([Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv14), attn_1], axis=3)  
        
        conv15 = self.__resnet_block(up15, 256, strides=1) 
        
        gating = self.UnetGatingSignal(conv15)
        attn_2 = self.AttnGatingBlock(conv12, gating, 128)
        up16 = concatenate([Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv15), attn_2], axis=3) 
        
        conv16 = self.__resnet_block(up16, 128, strides=1)      

        
        gating = self.UnetGatingSignal(conv16)
        attn_3 = self.AttnGatingBlock(conv11, gating, 64)
        up17 = concatenate([Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv16), attn_3], axis=3) 
        
        conv17 = self.__resnet_block(up17, 64, strides=1)   
   
        
        gating = self.UnetGatingSignal(conv17)
        attn_4 = self.AttnGatingBlock(conv10, gating, 32)
        up18 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv17), attn_4], axis=3) 
        
        conv18 = self.__resnet_block(up18, 32, strides=1)    
    
      #  conv18 = self.aspp_block(conv18, 32)
        
        conv19 = Conv2D(1, (1, 1), activation='sigmoid')(conv18)
    
        model = Model(inputs=[input_img], outputs=[conv19])

    

                        
        return model
    def aspp_block(self,x, filters):
     x1 = Conv2D(filters, (3, 3), dilation_rate=(6 * 1, 6 * 1), padding="same")(x)
     x1 = BatchNormalization()(x1)
     x2 = Conv2D(filters, (3, 3), dilation_rate=(12 * 1, 12 * 1), padding="same")(x)
     x2 = BatchNormalization()(x2)
     x3 = Conv2D(filters, (3, 3), dilation_rate=(18 * 1, 18 * 1), padding="same")(x)
     x3 = BatchNormalization()(x3)
     x4 = Conv2D(filters, (3, 3), padding="same")(x)
     x4 = BatchNormalization()(x4)
     y = Add()([x1, x2, x3, x4])
     y = Conv2D(filters, (1, 1), padding="same")(y)
     return y
    
    
    def expend_as(self,tensor, rep):
     my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
     return my_repeat
    def AttnGatingBlock(self,x, g, inter_shape):
        shape_x = K.int_shape(x)  # 32
        shape_g = K.int_shape(g)
        theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
        shape_theta_x = K.int_shape(theta_x)
        phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
        upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)  # 16
        concat_xg = add([upsample_g, theta_x])
        act_xg = Activation('relu')(concat_xg)
        psi = Conv2D(1, (1, 1), padding='same')(act_xg)
        sigmoid_xg = Activation('sigmoid')(psi)
        shape_sigmoid = K.int_shape(sigmoid_xg)
        upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32
        upsample_psi = self.expend_as(upsample_psi, shape_x[3])
        y = multiply([upsample_psi, x])
        result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
        result_bn = BatchNormalization()(result) 
        return result_bn
    
    def UnetGatingSignal(self,inputs):
     shape = K.int_shape(inputs)
     x = Conv2D(shape[3] * 2, (1, 1), strides=(1, 1), padding="same")(inputs)
     x = BatchNormalization()(x)
     x = Activation('relu')(x)
     return x
    def  __squeeze_excite_block(self,inputs, ratio=8):
       init = inputs
       channel_axis = -1
       filters = init.shape[channel_axis]
       se_shape = (1, 1, filters)

       se = GlobalAveragePooling2D()(init)
       se = Reshape(se_shape)(se)
       se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
       se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

       x = Multiply()([init, se])
       return x
    def __resnet_block(self,x, filters, strides=1):
     x_init = x
    ## Conv 1
     x = BatchNormalization()(x)
     x = Activation("relu")(x)
     x = Conv2D(filters, (3, 3), padding="same", strides=strides)(x)
    ## Conv 2
     x = BatchNormalization()(x)
     x = Activation("relu")(x)
     x = Conv2D(filters, (3, 3), padding="same", strides=1)(x)
    ## Shortcut
     s  = Conv2D(filters, (1, 1), padding="same", strides=strides)(x_init)
     s = BatchNormalization()(s)
   ## Add
     x = Add()([x, s])
     x =self. __squeeze_excite_block(x)
     return x
 

    
    
    # WARNING:tensorflow:AutoGraph could not transform <bound method UNetPlusPlus.__dice_coef_loss of 
    # <UNetPP.UNetPlusPlus object at 0x000001A33B0D8198>> and will run it as-is.
    # Cause: mangled names are not yet supported. To silence this warning, decorate the function with 
    # @tf.autograph.experimental.do_not_convert
    @tf.autograph.experimental.do_not_convert
    def __dice_coef(self, y_true, y_pred):
        
        """
        computes the dice loss. loss function for image segmentation 
        tasks is based on the Dice coefficient, which is essentially 
        a measure of overlap between two samples. This measure ranges 
        from 0 to 1 where a Dice coefficient of 1 denotes perfect and 
        complete overlap.
        
        Args:
            y_true: the true value of the image mask.
            y_pred: the predicted value of the image mask.
        
        Returns:
            dice_val: the dice loss value
            
        Ref:
            https://www.programmersought.com/article/11533881518/
            
        """
        
        y_true_f = K.flatten(y_true) # Extend y_true to one dimension.
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + self.__smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + self.__smooth)
    
    @tf.autograph.experimental.do_not_convert
    def __dice_coef_loss(self, y_true, y_pred):
        
        """
        computes the dice loss. loss function for image segmentation 
        tasks is based on the Dice coefficient, which is essentially 
        a measure of overlap between two samples. This measure ranges 
        from 0 to 1 where a Dice coefficient of 1 denotes perfect and 
        complete overlap.
        
        Args:
            y_true: the true value of the image mask.
            y_pred: the predicted value of the image mask.
        
        Returns:
            dice_val: the dice loss value
            
        Ref:
            https://www.programmersought.com/article/11533881518/
            
        """
       
        return 1. - self.__dice_coef(y_true, y_pred)
    
    
    def __loss(self,y_true, y_pred):
            return -(0.4*self.__dice_coef(y_true, y_pred)+0.6*self.__iou_loss_core( y_true, y_pred))

    
    @tf.autograph.experimental.do_not_convert
    def __iou_loss_core(self, y_true, y_pred):
        
        """
        computes the intersection over union metric. 
        Intersection over Union is an evaluation metric 
        used to measure the accuracy of an object/mask detected. 
        
        Args:
            y_true: the true value of the image mask.
            y_pred: the predicted value of the image mask.
            smooth: 
        
        Returns:
            iou: the iou coefficient.
            
        """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
        iou = (intersection + self.__smooth) / ( union + self.__smooth)
        
        return iou
    
    def __precision(self, y_true, y_pred,epsilon=1e-6):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    # Computing Sensitivity      
    def __sensitivity(self,y_true, y_pred,epsilon=1e-6):
     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
     return true_positives / (possible_positives + K.epsilon())
    
    def CompileAndSummarizeModel(self, model, optimizer = Adam(learning_rate=1e-4), loss = __dice_coef_loss):

        """
        Compiles and displays the model summary of the Unet++ model.

        Args:
            model: The keras instance of the Unet++ model.
            optimizer: model optimizer. Default is the adam optimizer.
            loss: the loss function. Default is the binary cross entropy loss.

        Return:
            None

        """
        model.compile(optimizer = optimizer, 
                      loss =self.__dice_coef_loss, 
                      metrics = ['acc', self.__iou_loss_core, self.__dice_coef,self.__precision,
                                 self.__sensitivity])
        
        model.summary()

    def plotModel(self, model, to_file = 'unetpp.png', show_shapes = True, dpi = 96):

        """
        Saves the Unet++ model plot/figure to a file.

        Args:
            model: The keras instance of the Unet++ model.
            to_file: the file name to save the model. Default name - 'unet.png'.
            show_shapes: whether to display shape information. Default = True.
            dpi: dots per inch. Default value is 96.

        Return:
            None

        """

        tf.keras.utils.plot_model(model, to_file = to_file, show_shapes = show_shapes, dpi = dpi)