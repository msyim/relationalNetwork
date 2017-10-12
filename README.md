# Relational Network
A TF implementation of the paper : ["A simple neural network module for relational reasoning"](https://arxiv.org/pdf/1706.01427.pdf)

## Architecture
image source: https://cdn-images-1.medium.com/max/2000/1*Fzkj-lSVmCGkptOwfbwUnA.png
![](https://cdn-images-1.medium.com/max/2000/1*Fzkj-lSVmCGkptOwfbwUnA.png)

## Modifications
Instead of using a learnable lookup table mentioned in the paper, I simply fed one-hot vectorized question words directly into LSTM.  Also, no data augmentation ( in the paper, input images are downsampled to 128 X 128, padded to 136 X 136 and cropped randomly back to 128 X 128 with light rotations from -.05 to .05 rads ) was performed.
