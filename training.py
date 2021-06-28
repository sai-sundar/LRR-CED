import tensorflow as tf
from models.dense import *
from models.unet import *
import numpy as np
from time import time
from scipy.misc import imsave
import pydicom
import glob
import math
from skimage.transform import radon, rescale, resize, iradon
from random import randint
import pandas as pd
from PIL import Image

h=512   # Image Height 
w=512   # Image Width
seed=1
m=64    # Size of Concatenation 1
n=128   # Size of Concatenation 2

# Normalisation

def normalise(X):

    return (X-np.amin(X))/(np.amax(X)-np.amin(X)+1e-4)

# Data Pre-processing:
data_dir = "/path/to/data/"

# Training data
GT_train = sorted(glob.glob(data_dir+"train/GT/*"))
sino_train = sorted(glob.glob(data_dir+"sino/*"))
bp_train = sorted(glob.glob(data_dir+"bp/*"))

# Valiadation data
GT_valid = sorted(glob.glob(data_dir+"validation/GT/*"))
sino_valid = sorted(glob.glob(data_dir+"validation/sino/*"))
bp_valid = sorted(glob.glob(data_dir+"validation/bp/*"))

# Test data
GT_test = sorted(glob.glob(data_dir+"test/GT/*"))
sino_test = sorted(glob.glob(data_dir+"test/sino/*"))
bp_test = sorted(glob.glob(data_dir+"test/bp/*"))

# Training data Generator
def generate_data(batch_size):
    """Replaces Keras' native ImageDataGenerator."""
    i = 0
    #file_list = os.listdir(directory)
    while True:
        x1 = []
        x2 = []
        x3 = []
        y = []
        for b in range(batch_size):
            value = randint(1, len(GT_train))           
            y.append(normalise(resize(np.asarray(Image.open(GT_dir1[value])),(512,512))))
            temp = resize(np.asarray(Image.open(sino_dir1[value])),(512,512))
            x1.append(normalise(temp))
            x2.append(normalise(resize(np.asarray(Image.open(bp_dir1[value])),(m,n))))
            x3.append(normalise(resize(np.asarray(Image.open(bp_dir1[value])),(256,256))))
            
            

        yield [(np.array(x1)).reshape((batch_size,512,512,1)),(np.array(x2)).reshape((batch_size,m,n,1)),(np.array(x3)).reshape((batch_size,256,256,1))],(np.array(y)).reshape((batch_size,512,512,1))

# Validation data Generator 
def valid_generator():
    """Replaces Keras' native ImageDataGenerator."""
    while True:
        
        value = randint(1,len(GT_valid))           
        y=normalise(resize(np.asarray(Image.open(GT_dir2[value])),(512,512)))
        temp = resize(np.asarray(Image.open(sino_dir2[value])),(512,512))
        x1=(normalise(temp))
        x2=(resize(np.asarray(Image.open(bp_dir2[value])),(m,n)))    
        x3=(normalise(resize(np.asarray(Image.open(bp_dir2[value])),(256,256))))    

        yield [x1.reshape((1,512,512,1)),x2.reshape((1,m,n,1)),x3.reshape((1,256,256,1))],y.reshape((1,512,512,1))


# Test data Generator
def test_generator(index):
    """Replaces Keras' native ImageDataGenerator."""
    
    value = index
    y=normalise(resize(np.asarray(Image.open(GT_dir3[value])),(512,512)))
    temp = resize(np.asarray(Image.open(sino_dir3[value])),(512,512))
    x1=(normalise(temp))
    x2=normalise(resize(np.asarray(Image.open(bp_dir3[value])),(m,m)))    
    x3=(normalise(resize(np.asarray(Image.open(bp_dir3[value])),(256,256))))            

    return x1.reshape((1,512,512,1)),x2.reshape((1,m,m,1)),x3.reshape((1,256,256,1)),y.reshape((1,512,512,1))


# Training 
# Create Folders to store checkpoints, reconstructed image predictions
num_epochs = 25      # Total number of epochs for training
sample_interval = 5  # Saving interval weights/predictions 

P_CED = P_CED_Dense(input_shape=(h,w,1)) 

# To define U_Net based P-CED 
# P_CED = P_CED_Unet(input_sizee=(h,w,1))  

for epoch in range(epochs):

    history = P_CED.fit_generator(generate_data(batch_size),steps_per_epoch = len(GT_dir)/batch_size,
                                                    validation_data = valid_generator(),validation_steps= len(GT_valid) ,  
                                                    epochs=1,shuffle=False,use_multiprocessing=True)
    # Storing metrics evaluated after every epoch    
    hist_df = pd.DataFrame(history.history) 
    hist_csv_file = 'History_Model_1.csv'
    with open(hist_csv_file, mode='a') as f:
        hist_df.to_csv(f)
    
    if epoch % sample_interval == 0:
                                
        P_CED.save_weights("checkpoints/gan_915_"+str(epoch)+".h5")        
        for i in range(1,5):
                    
            sino,bp,bp2,data=test_generator(i)
            img=P_CED.predict([sino.reshape((1,h,w,1)),bp.reshape((1,m,m,1)),bp2.reshape((1,256,256,1))])
            imsave(str(epoch)+"/image1_at_"+str(i)+".png",img.reshape(h,w))  
            imsave(str(epoch)+"/GT1_at_"+str(i)+".png",data.reshape(h,w))  
            imsave(str(epoch)+"/bp1_at_"+str(i)+".png",bp.reshape(m,m))   
        
        

    








