#!/usr/bin/env python

# python train_facade.py -g 0 -i ./facade/base --out result_facade --snapshot_interval 10000

from __future__ import print_function
import argparse
import os

import zipfile, shutil
from PIL import Image
import cv2

import chainer
from chainer.serializers.npz import NpzDeserializer
import numpy as np


from net import Encoder
from net import Decoder



ENC_W = "trained_model/enc_iter_176000.npz"
#DEC_W = "trained_model/dec_iter_176000.npz"
# to avoid GitHub 100M limit, one .npz files are divided into two zip files.
DEC_Ws = ["trained_model/dec_iter_176000.npz0","trained_model/dec_iter_176000.npz1"]

def loadimg(imgpath, min_wh=256):
    img = Image.open(imgpath)
    w,h = img.size
    r = min_wh/min(w,h)
    # resize images so that min(w, h) == min_wh
    img = img.resize((int(r*w), int(r*h)), Image.BILINEAR)
    img = np.asarray(img).astype("f").transpose(2,0,1)/128.0-1.0
    return img[:,0:256,0:256] # crop so that the shape be (3, 256, 256)
    
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
    # to avoid GitHub 100M limit, merge two files to restore the .npz file
    # if not os.path.exists(DEC_W):
        # shutil.copy(DEC_Ws[0], DEC_W)
        # with zipfile.ZipFile(DEC_W, 'a') as zp0:
            # zp1 = zipfile.ZipFile(DEC_Ws[1], 'r')
            # print(zp1.namelist())
            # for n in zp1.namelist():
                # zp0.writestr(n, zp1.open(n).read())
    for npzfile in DEC_Ws:
        with np.load(npzfile) as f:
            d = NpzDeserializer(f, strict=False)
            d.load(dec)
    
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

    if args.gpu >= 0:
        outimg = x_out.data.get()[0]
    else:
        outimg = x_out.data[0]
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
