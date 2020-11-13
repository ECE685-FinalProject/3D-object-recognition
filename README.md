# Two-stage 3D-object-recognition

Jiawei Chen, Linlin Li


This repo is a PyTorch implementation for **two-stage PointConv for Classification on Point Clouds** adapting from **PointConv**(https://arxiv.org/abs/1811.07246). Our code skeleton is borrowed from [DylanWusee/pointconv_pytorch](https://github.com/DylanWusee/pointconv_pytorch).

Nonnegligible differences in the prediction accuracy of different objects are reported in 3D object classification. For example, Garcia et al (https://ieeexplore.ieee.org/document/7727386) realized that classifiers, especially those CNN-based classifiers, were confused on objects that look alike, such as desk and table. Due to the nature of the CNNs, which is heavily dependent on the combinations of features, this type of errors are common. Rather than directly training a classifier to distinguish between all classes of objects, a two-stage classifier is proposed to improve prediction accuracy, especially for objects with similar appearance.

## Installation
The code is modified from repo [PointConv_pytorch](https://github.com/DylanWusee/pointconv_pytorch). Please install [PyTorch](https://pytorch.org/), [pandas](https://pandas.pydata.org/), and [sklearn](https://scikit-learn.org/).
The code has been tested with Python 3.7, pytorch 1.6, CUDA 10.2.

## Usage
### ModelNet40 Classification

Download the ModelNet40 dataset from [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip). This dataset is the same one used in [PointNet](https://arxiv.org/abs/1612.00593), thanks to [Charles Qi](https://github.com/charlesq34/pointnet). Copy the unziped dataset to ```./data/ModelNet```. 

To train the $K-1$ classifier,
```
python kb1_train_cls_conv.py
```

To train the binary classifier,
```
python binary_train_cls_conv.py
```

To evaluate the two-stage classifier,
```
python k_eval_cls_conv.py --checkpoint=/path/to/checkpoint
```
