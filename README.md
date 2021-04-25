# CRISPR-ONT & CRISPR-OFFT

## Overview
CRISPR-ONT and CRISPR-OFFT are attention-based convolution neural networks (CNNs) frameworks to accurately predict CRISPR/Cas9 sgRNA on- and off-target activities, respectively. By integrating CNNs with attention mechanism, both CRISPR-ONT and CRISPR-OFFT are able to capture the intrinsic characteristics of Cas9-sgRNA binding and cleavage, thus improving the accuracy and interpretability.

## Dependencies  
* **[Ubuntu](https://www.ubuntu.com/download/desktop) 16.04 or later**
* **[Anaconda](https://www.anaconda.com/distribution/#download-section) 3-5.2.0 or later**
* **Python packages:**   
  [numpy](https://numpy.org/)   
  [pandas](https://pandas.pydata.org/)      
 * **[Keras](https://keras.io/)**    
 * **[Tensorflow](https://tensorflow.google.cn/)**   

## File Description  
* crispr_ont.h5: the weights for the CRISPR-ONT model  
* crispr_offt.h5: the weights for the CRISPR-OFFT model  
* crispr_on_prediction.py: CRISPR-ONT code for sgRNA on-target activity prediction    
* crispr_offt_prediction.py: CRISPR-OFFT code for sgRNA off-target activity prediction     
  
## Testing CRISPR-ONT and CRISPR-OFFT with test set 
python crispr_on_prediction.py (for on-target activity prediction)      
python crispr_offt_prediction.py (for off-target activity prediction)      

## Docker  
Alternatively, you can start a Docker container and exec into it.  
```
sudo docker build -t crispr .  
sudo docker run crispr  
``` 

