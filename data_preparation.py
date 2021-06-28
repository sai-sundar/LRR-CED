import glob
import astra
import os
import math
import numpy as np
import pydicom
from PIL import Image


N_A = 60  # Number of views in the projections
N_D = 700 # Number of detectors


#ASTRA Initialization
vol_geom = astra.create_vol_geom(512, 512)                                                             # Volume 
proj_geom1 = astra.create_proj_geom('fanflat', 1.0, 700, np.linspace(0,2*np.pi,NUM_OF_VIEWS,False), -1500,500)  # Projector geometry
rec_id = astra.data2d.create('-vol', vol_geom)                                                         # Recontruction Volume Initialization 
proj_id1 =  astra.create_projector('line_fanflat',proj_geom1,vol_geom)                                 # Projector 


# Function to convert HUT to Attenuation(mm-1)

def attenuation(X):
    
    atten = np.zeros((512,512))
    #X = resize(X, (128,128))
    for i in range(512):

        for j in range(512):
            if X[i][j] < -1000:
                atten[i][j] = 0.015
            elif X[i][j] < 0:
                atten[i][j] = 0.01*(0.15 + 0.00015*X[i][j])
            else:
                atten[i][j] = 0.01*(0.15 + 7.04*0.00015*X[i][j])
         
       
    return atten


count=0
data_path=[]
# Dataset can be downloaded from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70224216#70224216bcab02c187174a288dbcbf95d26179e8

data_path = sorted(glob.glob("/path/to/dataset/Lung-PET-CT-Dx/*/*/*/"))   
save_dir = "/path/to/save/data/"

N_P = 200 # Number of patients to process

for i in range(1,400):
    
    
    editFiles = []
    
    editFiles=(os.listdir(data_path[i]+'/*'))
    
       
    sorted_Files = sorted(editFiles)
    
    for files1 in range(len(sorted_Files)):
              
        
        dcm  = pydicom.dcmread(sorted_Files[files1]) 
        
        if dcm[0x0008,0x0060].value =='PT':
            print("PET"+str(i))
            break
        temp = dcm.pixel_array
        try:
            atten = attenuation(temp.reshape(512,512))
        except ValueError:
            break
        
        if math.isnan(np.amax(atten)):         
            break
        
        sinogram_id1, sinogram = astra.create_sino(atten, proj_id1)          # Creating Sinograms      
        sino_N1 = astra.functions.add_noise_to_sino(sinogram, 1e5)           # Adding Poisson Noise to sinograms
        sN_id1 = astra.data2d.create("-sino",proj_geom1,sino_N1)               
        
                                             
        # Reconstructing images with FBP

        cfg = astra.astra_dict('FBP')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sN_id1
        cfg['ProjectorId'] = proj_id1
        cfg['option'] = { 'FilterType': 'Ram-Lak' }


        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        rec = astra.data2d.get(rec_id)


        # Saving the Sinogram, GT and FBP reconstructed images        
        im = Image.fromarray(sino_N1)
        im.save(save_dir+"sino/"+str(i)+"_"+str(files1)+".tiff")
        im = Image.fromarray(rec.reshape(512,512))
        im.save(save_dir+"bp/"+str(i)+"_"+str(files1)+".tiff")
        im = Image.fromarray(atten.reshape(512,512))
        im.save(save_dir+"GT/"+str(i)+"_"+str(files1)+".tiff")
        

# Deleting ASTRA variables        
astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)
astra.data2d.delete(sinogram_id1)
astra.projector.delete(proj_id1)
astra.data2d.delete(sN_id1)
