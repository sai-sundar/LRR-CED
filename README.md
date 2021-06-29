## LRR-CED: Low Resolution Reconstruction aware Convolutional Encoder-Decoder Network for Direct Sparse-View CT Image Reconstruction 

This work provides a framework for sparse-view CT image reconstruction using fully convolutional encoder-decoder networks. We provide two variations of the approach namely LRR-CED (D) with fully convolutional dense networks and LRR-CED(U) with UNet. 

# Requirements 
tensorflow 2.4__
pydicom__
astra toolbox

# Repo Description 

*demo.ipynb*  contains an out of the box demo that load the weights and demonstrates results on test data 
*data_preparation.py* script for data preparationd and pre-processing required for training the neural networks from scratch
*training.py* script for training the neural networks

# References 

[1] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
[2] Jégou, Simon, et al. "The one hundred layers tiramisu: Fully convolutional densenets for semantic segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2017.
[3] Van Aarle, Wim, et al. "Fast and flexible X-ray tomography using the ASTRA toolbox." Optics express 24.22 (2016): 25129-25147.
