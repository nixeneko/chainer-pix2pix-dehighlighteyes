import os

import numpy
from PIL import Image
import six

import numpy as np

from io import BytesIO
import os
import pickle
import json
import numpy as np
import glob

import skimage.io as io

from chainer.dataset import dataset_mixin

# download `BASE` dataset from http://cmp.felk.cvut.cz/~tylecr1/facade/
class ImgDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataSrcDir, dataDstDir, data_range=(0.0,0.8)):
        print("load dataset start")
        print("    from: %s, %s"%(dataSrcDir, dataDstDir))
        print("    range: [{}, {})".format(data_range[0], data_range[1]))
        self.dataSrcDir = dataSrcDir
        self.dataDstDir = dataDstDir
        self.dataset = []
        self.picfiles = list(map(os.path.basename, glob.glob(os.path.join(dataDstDir, "*.jpg"))))
        data_range_start = int(data_range[0] * len(self.picfiles))
        data_range_end   = int(data_range[1] * len(self.picfiles))
        for fn in self.picfiles[data_range_start:data_range_end]:
            img_src = Image.open(os.path.join(self.dataSrcDir, fn))
            img_dst = Image.open(os.path.join(self.dataDstDir, fn))
            w,h = img_src.size
            r = 286/min(w,h)
            # resize images so that min(w, h) == 286
            img_src = img_src.resize((int(r*w), int(r*h)), Image.BILINEAR)
            img_dst = img_dst.resize((int(r*w), int(r*h)), Image.BILINEAR)
            
            img_src = np.asarray(img_src).astype("f").transpose(2,0,1)/128.0-1.0
            img_dst = np.asarray(img_dst).astype("f").transpose(2,0,1)/128.0-1.0
            self.dataset.append((img_src, img_dst))
        print("load dataset done")
    
    def __len__(self):
        return len(self.dataset)

    # return (label, img)
    def get_example(self, i, crop_width=256):
        _,h,w = self.dataset[i][0].shape
        x_l = np.random.randint(0,w-crop_width)
        x_r = x_l+crop_width
        y_l = np.random.randint(0,h-crop_width)
        y_r = y_l+crop_width
        #same image for input and output image pair
        return self.dataset[i][0][:,y_l:y_r,x_l:x_r],self.dataset[i][1][:,y_l:y_r,x_l:x_r]
    
