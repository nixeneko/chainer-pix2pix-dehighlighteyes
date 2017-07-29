#!/usr/bin/env python

# python train_facade.py -g 0 -i ./facade/base --out result_facade --snapshot_interval 10000

from __future__ import print_function
import argparse
import os

import chainer
import numpy as np

from net import Encoder
from net import Decoder

from PIL import Image
import cv2

ENC_W = "trained_model/enc_iter_176000.npz"
DEC_W = "trained_model/dec_iter_176000.npz"

def loadimg(imgpath, min_wh=256):
    img = Image.open(imgpath)
    w,h = img.size
    r = min_wh/min(w,h)
    # resize images so that min(w, h) == min_wh
    img = img.resize((int(r*w), int(r*h)), Image.BILINEAR)
    img = np.asarray(img).astype("f").transpose(2,0,1)/128.0-1.0
    return img
    
def to_bgr(ary):
    print(ary.shape)
    bgr = np.asarray(ary)[::-1]
    print(bgr.shape)
    bgrimg = bgr.transpose(1,2,0) / 2 + 0.5 # [0, 1]
    print(bgrimg.shape)
    #bgrimgint = (bgrimg * 128.0 + 128.0).astype("i")
    
    return bgrimg
    #((np.asarray(ary)[::-1].transpose(1,2,0) + 1.0) * 128.0).astype("i")
    
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
    chainer.serializers.load_npz(DEC_W, dec)
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        enc.to_gpu()  # Copy the model to the GPU
        dec.to_gpu()

    inimg = loadimg(args.img)
    x_in = inimg[np.newaxis,:]
    print(x_in.shape)
    # x_in as an input image
    x_in = chainer.Variable(x_in)
    if args.gpu >= 0:
        x_in.to_gpu()
    z = enc(x_in)
    x_out = dec(z)

    outimg = x_out.data.get()[0]
    #img_show = np.zeros((inimg.shape[0], inimg.shape[1], inimg.shape[2]*2))
    #img_show[:,:,:inimg.shape[2]] = inimg
    #img_show[:,:outimg.shape[1],inimg.shape[2]:inimg.shape[2]+outimg.shape[2]] = outimg
    img_show = np.concatenate((inimg, outimg), axis=2)
    bgrpic = to_bgr(img_show).copy()
    cv2.putText(bgrpic,"input",(3,15),cv2.FONT_HERSHEY_DUPLEX, 0.5,(255,0,0))
    cv2.putText(bgrpic,"output",(259,15),cv2.FONT_HERSHEY_DUPLEX, 0.5,(255,0,0))
    cv2.imshow("result", bgrpic)
    cv2.waitKey()
    
if __name__ == '__main__':
    main()
