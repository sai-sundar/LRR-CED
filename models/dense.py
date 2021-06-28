import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda, Add
import numpy as np
from tensorflow.keras.layers import Activation,MaxPooling2D,UpSampling2D,Dense,BatchNormalization,Input,Reshape,multiply,add,Dropout,AveragePooling2D,GlobalAveragePooling2D,concatenate
from tensorflow.keras.layers import Conv2D,Conv2DTranspose
from tensorflow.keras.models import Model														  
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications.vgg16 import VGG16
from skimage.transform import iradon

def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2):
    
    l = BatchNormalization()(inputs)
    l = Activation('relu')(l)
    l = Conv2D(n_filters, filter_size, padding='same', kernel_initializer='he_uniform')(l)
    if dropout_p != 0.0:
        l = Dropout(dropout_p)(l)
    return l


def TransitionDown(inputs, n_filters, dropout_p=0.2):
    
    l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
    l = MaxPooling2D((2,2))(l)
    return l

def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
        
    l = Conv2DTranspose(n_filters_keep, kernel_size=3, strides=2, padding='same', kernel_initializer='he_uniform')(block_to_upsample)
    l = concatenate([l, skip_connection], axis=-1)
    return l

def SoftmaxLayer(stack, n_classes):
    
    l = Conv2D(n_classes, kernel_size=1, padding='same', kernel_initializer='he_uniform')(stack)
    l = Activation('sigmoid')(l)#or softmax for multi-class
    
    return l
    

# Custom Loss Function     
def perceptual_loss(y_true, y_pred):
    y_pred = tf.image.grayscale_to_rgb(y_pred)
    y_true = tf.image.grayscale_to_rgb(y_true)
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(128,128,3))
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    loss_model2 = Model(inputs=vgg.input, outputs=vgg.get_layer('block1_conv2').output)
    loss_model2.trainable = False
    
    return (0.5*K.mean(K.abs(loss_model(y_true) - loss_model(y_pred)))+10*K.mean(K.abs(y_true - y_pred)) + 0.5*K.mean(K.abs(loss_model2(y_true) - loss_model2(y_pred))))

optimizer = Adam(0.0002, 0.5)

def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    y_pred = K.clip(y_pred, 0.0, 1.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))

def SSIM(y_true,y_pred):

    return tf.image.ssim(y_true,y_pred,max_val=1.0)

def LRRCED_D(
        input_shape=(None,None,3),
        n_classes = 1,
        n_filters_first_conv = 32,
        n_pool = 5,
        growth_rate = 16,
        n_layers_per_block = [4,5,7,10,12,15,12,10,7,5,4],
        dropout_p = 0.0
        ):
    if type(n_layers_per_block) == list:
        print(len(n_layers_per_block))
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError
        
#####################
# First Convolution #
#####################        
    inputs = Input(shape=input_shape)
    input2 = Input(shape=(64,64,1))    # Concatenation 1
    input3 = Input(shape=(128,128,1))  # Concatenation 2
    stack = Conv2D(filters=n_filters_first_conv, kernel_size=3, padding='same', kernel_initializer='he_uniform')(inputs)
    n_filters = n_filters_first_conv

#####################
# Downsampling path #
#####################     
    skip_connection_list = []
    
    for i in range(n_pool):
        for j in range(n_layers_per_block[i]):
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            stack = concatenate([stack, l])
            n_filters += growth_rate
        
        skip_connection_list.append(stack)        
        stack = TransitionDown(stack, n_filters, dropout_p)
    skip_connection_list = skip_connection_list[::-1]

    
#####################
#    Bottleneck     #
#####################     
    block_to_upsample=[]
    
    for j in range(n_layers_per_block[n_pool]):
        l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
        block_to_upsample.append(l)
        stack = concatenate([stack,l])
    block_to_upsample = concatenate(block_to_upsample)
    #block_to_upsample = concatenate([block_to_upsample,input2],axis=3)
   
#####################
#  Upsampling path  #
#####################
    for i in range(n_pool):
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i ]
        stack = TransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep)
        
        block_to_upsample = []
        for j in range(n_layers_per_block[ n_pool + i + 1 ]):
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            block_to_upsample.append(l)
            stack = concatenate([stack, l])
        block_to_upsample = concatenate(block_to_upsample)
        if i==1:                                                                 # Concatenating at the appropriate levels
            block_to_upsample = concatenate([block_to_upsample,input2],axis=3)
        if i==2:
            block_to_upsample = concatenate([block_to_upsample,input3],axis=3)
         
#####################
#  Softmax          #
#####################
     
    output = SoftmaxLayer(stack, n_classes)            
    model=Model(inputs = [inputs,input2,input3], outputs = output)    
    model.compile(loss=perceptual_loss, optimizer =   optimizer, metrics=[SSIM,PSNR])
    #model.summary()
    return model
    


    
