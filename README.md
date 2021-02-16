# CRISPR-ONT & CRISPR-OFFT

## Overview
CRISPR-ONT and CRISPR-OFFT are attention-based convolution neural networks (CNNs) frameworks to accurately predict CRISPR/Cas9 sgRNA on- and off-target activities, respectively. By integrating the CNNs with attention mechanism, both CRISPR-ONT and CRISPR-OFFT are able to capture the intrinsic characteristics of Cas9-sgRNA binding and cleavage, thus improving the accuracy and interpretability.

## Pre-requisite:  
* **[Ubuntu](https://www.ubuntu.com/download/desktop) 16.04 or later**
* **Anaconda 3-5.2.0 or later**
* **Python packages:**   
  [numpy](https://numpy.org/) 
  [pandas](https://pandas.pydata.org/) 
  [scikit-learn](https://scikit-learn.org/stable/)  
  [scipy](https://www.scipy.org/)  
 * **[Keras](https://keras.io/)**    
 * **[Tensorflow].https://tensorflow.google.cn/)**   
   
  
## Installation guide
#### **Operation system**  
Ubuntu download from https://www.ubuntu.com/download/desktop  
#### **Python and packages**  
Download Anaconda tarball on https://www.anaconda.com/distribution/#download-section  
#### **Tensorflow installation:**  
pip install tensorflow-gpu==1.4.0 (for GPU use)  
pip install tensorflow==1.4.0 (for CPU use)  
#### **CUDA toolkit 8.0 (for GPU use)**     
Download CUDA tarball on https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run  
#### **cuDNN 6.1.10 (for GPU use)**      
Download cuDNN tarball on https://developer.nvidia.com/cudnn  
 
## Content
* **./data/test.csv:** The testing examples with sgRNA sequence and label indicating the on-target cleavage efficacy  
* **./weights/CRISPR-ONT.h5:** The well-trained weights for our CRISPR-ONT model
* **./CRISPR-ONT.py:** The python code of CRISPR-ONT model, it can be ran to reproduce our results
* **./CRISPR-OFFT.py:** The python code of CRISPR-OFFT model, it can be ran to reproduce our results
* **./result/result.csv:** The prediction results of A-CNNCrispr model

## Usage
## Testing CRISPR-ONT with test set
#### **python CRISPR-ONT.py** 

## Testing CRISPR-OFFT with test set
#### **python CRISPR-OFFT.py** 
