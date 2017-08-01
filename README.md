# chainer-pix2pix-dehighlighteyes
This is a modified version of the Chainer implementation of pix2pix (chainer-pix2pix) which is available on:
 - https://github.com/pfnet-research/chainer-pix2pix

This modified version enables to train on datasets with any RGB color image pairs instead of the Facade dataset.

## Trained model
This repository contains a trained model which was trained on the dataset which contains pictures of animation characters as input and corresponding character face images with no highlight on eyes as target (ground truth).

The model is trained with 450 pairs of images as an training set and 50 pairs as an verification set.

# Example result of the trained model
<img src="https://raw.githubusercontent.com/nixeneko/chainer-pix2pix-dehighlighteyes/master/example.jpg">

# Usage
## Demo
`demo.py` loads the trained model, does the transformation on any given image and shows the result.

1. Install Chainer, CuPy (for GPU support, not mandatory), OpenCV 3 python bindings (to show the result).
2. `python demo.py -g <GPU number> -i <picture file>`
  - A face area that has the size around 256x256 px can give better result, since the model is trained on 256x256 face images.

## Training
1. Install Chainer (developed on ver. 2.0.1) and CuPy (for GPU support, not mandatory but highly recommended. developed on ver. 1.0.1).
2. Put source images (input) and target images (ground truth) in separate directories. <!-- and specify the paths in `train_dehighlight.py`.-->
3. `python train_facade.py --gpu <GPU number> --data_src <dataset source dir> --data_dst <dataset target dir> --out <output dir>`
4. Wait several hours. 
  - Trained modeles and visualized results are put in the directory specified in `--out` every number of iteration specified in `--snapshot_interval`.
  - Since models are large, smaller number of `--snapshot_interval` can cause disk full.

 
