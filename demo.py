#!/usr/bin/env python

# python train_facade.py -g 0 -i ./facade/base --out result_facade --snapshot_interval 10000

from __future__ import print_function
import argparse
import os

import math
from PIL import Image
import cv2

import chainer
from chainer.serializers.npz import NpzDeserializer
import numpy as np


from net import Encoder
from net import Decoder

# trained models
ENC_W = "trained_model/enc_iter_176000.npz"
#DEC_W = "trained_model/dec_iter_176000.npz"
# to avoid GitHub 100M limit, one .npz files are divided into two zip files.
DEC_Ws = ["trained_model/dec_iter_176000.npz0","trained_model/dec_iter_176000.npz1"]

def loadimg(imgpath, min_wh=256):
    img = Image.open(imgpath)
    #w,h = img.size
    #r = min_wh/min(w,h)
    # resize images so that min(w, h) == min_wh
    #img = img.resize((int(r*w), int(r*h)), Image.BILINEAR)
    img = np.asarray(img).astype("f").transpose(2,0,1)/128.0-1.0
    return img
    
def to_bgr(ary):
    # input shape: (c, h, w)
    # print(ary.shape)
    bgr = np.asarray(ary)[::-1] # rgb to bgr
    # print(bgr.shape)
    bgrimg = bgr.transpose(1,2,0) / 2 + 0.5 # [0, 1]
    # print(bgrimg.shape)
    #bgrimgint = (bgrimg * 128.0 + 128.0).astype("i")
    
    return bgrimg
    
def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--img', '-i', help='Input image')
    parser.add_argument('--out', '-o', default='result_dehighlight',
                        help='Directory to output the result')
    args = parser.parse_args()

    
    # Set up a neural network to train
    enc = Encoder(in_ch=3)
    dec = Decoder(out_ch=3)
    
    chainer.serializers.load_npz(ENC_W, enc)
    # to avoid GitHub 100M limit, 1 .npz file is devided into 2 files
    for npzfile in DEC_Ws:
        with np.load(npzfile) as f:
            d = NpzDeserializer(f, strict=False)
            d.load(dec)
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        enc.to_gpu()  # Copy the model to the GPU
        dec.to_gpu()

    inimg = loadimg(args.img)
    ch, h, w = inimg.shape
    # add paddings so that input array has the size of mutiples of 256.
    in_ary = np.zeros((ch,math.ceil(h/256)*256, math.ceil(w/256)*256), dtype="f") 
    in_ary[:,0:h,0:w] = inimg
    x_in = in_ary[np.newaxis,:] # to fit into the minibatch shape
    print(x_in.shape)
    # x_in as an input image
    x_in = chainer.Variable(x_in)
    if args.gpu >= 0:
        x_in.to_gpu()
    z = enc(x_in)
    x_out = dec(z)

    if args.gpu >= 0:
        out_ary = x_out.data.get()[0]
    else:
        out_ary = x_out.data[0]
    #img_show = np.zeros((inimg.shape[0], inimg.shape[1], inimg.shape[2]*2))
    #img_show[:,:,:inimg.shape[2]] = inimg
    #img_show[:,:outimg.shape[1],inimg.shape[2]:inimg.shape[2]+outimg.shape[2]] = outimg
    outimg = out_ary[:,0:h,0:w] # trim paddings
    img_show = np.concatenate((inimg, outimg), axis=2)
    bgrpic = to_bgr(img_show).copy()
    cv2.putText(bgrpic,"input",(3,15),cv2.FONT_HERSHEY_DUPLEX, 0.5,(255,0,0))
    cv2.putText(bgrpic,"output",(w+3,15),cv2.FONT_HERSHEY_DUPLEX, 0.5,(255,0,0))
    cv2.imshow("result", bgrpic)
    cv2.waitKey()
    
if __name__ == '__main__':
    main()
