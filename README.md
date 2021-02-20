# CRISPR-ONT & CRISPR-OFFT

## Overview
CRISPR-ONT and CRISPR-OFFT are attention-based convolution neural networks (CNNs) frameworks to accurately predict CRISPR/Cas9 sgRNA on- and off-target activities, respectively. By integrating CNNs with attention mechanism, both CRISPR-ONT and CRISPR-OFFT are able to capture the intrinsic characteristics of Cas9-sgRNA binding and cleavage, thus improving the accuracy and interpretability.

## Dependencies:  
* **[Ubuntu](https://www.ubuntu.com/download/desktop) 16.04 or later**
* **[Anaconda](https://www.anaconda.com/distribution/#download-section) 3-5.2.0 or later**
* **Python packages:**   
  [numpy](https://numpy.org/)   
  [pandas](https://pandas.pydata.org/)   
  [scikit-learn](https://scikit-learn.org/stable/)       
 * **[Keras](https://keras.io/)**    
 * **[Tensorflow](https://tensorflow.google.cn/)**   

## Quickstart Guide: 
CRISPR-ONT takes the DNA sequence of the guide and PAM sequence (23 base pair sequence) as the input. No other input is required for this model. Here we detail the instruction to use the light-weight prediction tool by running the `CRISPR-ONT_prediction` script.

**Step 1:** the code asks for the sgRNA sequence followed by the PAM sequence.
```
Input the sgRNA sequence followed by the PAM sequence (23 base pair sequence):  
ACTGCATGCATCGACGCCCGGGG
```
**Step 2:** the CRISPR-ONT outputs the predicted results.
```
Here is the cleavage efficiency that CRISPR-ONT predicts for this guide:    
0.70
```
CRISPR-OFFT takes the sgRNA-DNA sequence pair with lenght of 23 as the inputs. Here we detail the instruction to use the light-weight prediction tool by running the `CRISPR-OFFT_prediction` script.
**Step 1:** the code asks for the sgRNA sequence followed by the PAM sequence.
```
Input the sgRNA sequence followed by the PAM sequence (23 base pair sequence):  
AAATGAGAAGAAGAGGCACAGGG
```
**Step 2:** the code asks for the DNA sequence.
```
Input the DNA sequence:  
GCATGAGAAGAAGAGACATAGCC
```
**Step 3:** the CRISPR-OFFT outputs the predicted result.
```
Predicting on test data:  
The input sequence belongs to non-off-target with possibility 1.0000
```
## Usage
### Testing CRISPR-ONT with test set  
* python CRISPR-ONT.py   

### Testing CRISPR-OFFT with test set
* python CRISPR-OFFT.py  


## Contact
We looking forward to receiving any bug reports and suggestions. If you have any questions, feel free to E-mail me via: `gszhang(at)stu.edu.cn`
