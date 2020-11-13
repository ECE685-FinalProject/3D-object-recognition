# Two-stage 3D-object-recognition

Jiawei Chen, Linlin Li


This repo is a PyTorch implementation for Dynamic Graph CNN for Learning on Point Clouds (DGCNN)(https://arxiv.xilesou.top/pdf/1801.07829). Our code skeleton is borrowed from WangYueFt/dgcnn.

Note that the network structure (Fig. 3) for classification in DGCNN paper is not consistent with the corresponding description in section 4.1 of the paper. The author of DGCNN adopts the setting of classification network in section 4.1, not Fig. 3. We fixed this mistake in Fig. 3 using PS and present the revised figure below.

Unlike images which are represented in regular dense grids, 3D point clouds are irregular and unordered, hence applying convolution on them can be difficult. In this paper, we extend the dynamic filter to a new convolution operation, named PointConv. PointConv can be applied on point clouds to build deep convolutional networks. We treat convolution kernels as nonlinear functions of the local coordinates of 3D points comprised of weight and density functions. With respect to a given point, the weight functions are learned with multi-layer perceptron networks and the density functions through kernel density estimation. A novel reformulation is proposed for efficiently computing the weight functions, which allowed us to dramatically scale up the network and significantly improve its performance. The learned convolution kernel can be used to compute translation-invariant and permutation-invariant convolution on any point set in the 3D space. Besides, PointConv can also be used as deconvolution operators to propagate features from a subsampled point cloud back to its original resolution. Experiments on ModelNet40, ShapeNet, and ScanNet show that deep convolutional neural networks built on PointConv are able to achieve state-of-the-art on challenging semantic segmentation benchmarks on 3D point clouds. Besides, our experiments converting CIFAR-10 into a point cloud showed that networks built on PointConv can match the performance of convolutional networks in 2D images of a similar structure.

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
