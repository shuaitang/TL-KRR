## Transfer Learning with Kernel Ridge Regression (TL-KRR)
---------------------------------------------------------

Here comes the implementation of our paper on transfer learning with kernel ridge regression, and it doesn't require finetuning the base model. 

In our experiments, we primarily tested on transferring from ResNet models pretrained on the ImageNet dataset to six downstream tasks, including CIFAR10, CIFAR100, STL10, CUB200, SVHN and Kuzushiji49.

The details of our method in four steps are presented in the [paper](https://arxiv.org/pdf/2006.06791.pdf):

*Transfer Learning with Kernel Ridge Regression*

by Shuai Tang and Virginia R. de Sa

## Brief Introduction
The Implementation relies on the following files:

*TLKRR.py* is the main file that conducts our four-step transfer learning method with kernel ridge regression.

*sketched_kernels.py* sketches the feature vectors at individual layers into a fixed number of buckets. It now supports feature hashing as the first step to reduce the dimensionality of feature vectors so that one could deploy deeper models.

*lowrank_feats.py* applies the Nyström method on top of feature vectors to compute low-rank approximations.

*learning_kernel_alignment.py* computes the optimal convex combination of feature vectors from individual layers that gives the highest alignment score with the target in a downstream task.

*utils.py* has helper functions.

*cub200.py* and *kuzushiji49.py* implement the PyTorch vision dataset class for CUB200 and Kuzushiji49, respectively.

## Requirements
```
python >= 3.5
torch >= 1.0
torchvision
numpy
scipy
sklearn
pandas
```

## Simple Example
```
CUDA_VISIBLE_DEVICES=0 python3 -u TLKRR.py \
    --datapath data/ \
    --modelname resnet18 \
    --task cifar100 \
    --bsize 800
```

To reduce the memory consumption, one could do the following:
```
CUDA_VISIBLE_DEVICES=0 python3 -u TLKRR.py \
    --datapath data/ \
    --modelname wide_resnet50_2 \
    --task cifar100 \
    --bsize 400 \
    --feature_hashing --factor 4
```
One could set *factor* or/and *M* to a large number to get decent performance.

## Authors  
Shuai Tang

## Acknowledgements
We gratefully thank [Charlie Dickens](https://c-dickens.github.io/) and [Wesley J. Maddox](https://wjmaddox.github.io/) for fruitful discussions, and appreciate [Richard Gao](http://www.rdgao.com/), [Mengting Wan](https://mengtingwan.github.io/) and [Shi Feng](http://users.umiacs.umd.edu/~shifeng/) for comments on the draft. 
Huge amount of thanks to my advisor --- [Virginia de Sa](http://www.cogsci.ucsd.edu/~desa/) --- for basically allowing me to do whatever I am interested in. :)  

## Rant
If the number of data samples is extremely small, then one should skip the first step of approximating low-rank features, otherwise the *undersampling* issue would occur and hurt the performance.

For memory concern, one could set the precision of generated feature vectors to half-precision floating-point, and it gives a minor performance drop.