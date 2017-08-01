#!pythonwin
# coding: utf-8

import matplotlib.pyplot as plot

import os, sys, glob, tempfile, shutil, io
import cv2
import numpy as np

import argparse
import os

#from PIL import Image

import chainer
from chainer.serializers.npz import NpzDeserializer


from net import Encoder
from net import Decoder
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,
                              encoding=sys.stdout.encoding, 
                              errors='backslashreplace', 
                              line_buffering=sys.stdout.line_buffering)

INFILE = "test.avi"
OUTFILE = "test_done.avi"

ESC_KEY = 27     # Escキー
#WAIT_INTERVAL= 1000//24     # 待ち時間
#FRAME_RATE = 30  # fps

WINNAME = "test"

# models
ENC_W = "trained_model/enc_iter_176000.npz"
DEC_Ws = ["trained_model/dec_iter_176000.npz0","trained_model/dec_iter_176000.npz1"]


    
def loadimg(img):
    img_resized = cv2.resize(img, (768, 432))
    
    img = np.asarray(img_resized).astype("f").transpose(2,0,1)/128.0-1.0
    img_rgb = img[::-1] #BGR to RGB
    return img_rgb
    # w,h = img.size
    # r = min_wh/min(w,h)
    # resize images so that min(w, h) == min_wh
    # img = img.resize((int(r*w), int(r*h)), Image.BILINEAR)
    # img = np.asarray(img).astype("f").transpose(2,0,1)/128.0-1.0
    # return img[:,0:256,0:256] # crop so that the shape be (3, 256, 256)
    
    
def to_bgr(ary):
    # input shape: (c, h, w)
    # print(ary.shape)
    bgr = np.asarray(ary)[::-1] # rgb to bgr
    # print(bgr.shape)
    bgrimg = bgr.transpose(1,2,0) / 2 + 0.5 # [0, 1] value w/ shape (h, w, c)
    # print(bgrimg.shape)
    #bgrimgint = (bgrimg * 128.0 + 128.0).astype("i")
    
    return bgrimg

class MovieIter(object):
    def __init__(self, moviefile):
        #TODO: check if moviefile exists
        self.org = cv2.VideoCapture(moviefile)
        # self.end_flg, self.frame_ultimate = self.org.read()
    def __iter__(self):
        return self
    def __next__(self):
        # while True:
            # self.frame_penultimate = self.frame_ultimate
            self.end_flg, self.frame_ultimate = self.org.read()
            if not self.end_flg: # end of the movie
                raise StopIteration()
            # diff = self.frame_ultimate - self.frame_penultimate
            # val = np.sum(np.abs(diff))
            # if val >= 10000000:  # ignore non-moving frames
            return self.frame_ultimate
        #org.release()

def main():
    cv2.namedWindow(WINNAME)
    
    # Set up a neural network
    enc = Encoder(in_ch=3)
    dec = Decoder(out_ch=3)
    
    chainer.serializers.load_npz(ENC_W, enc)
    for npzfile in DEC_Ws:
        with np.load(npzfile) as f:
            d = NpzDeserializer(f, strict=False)
            d.load(dec)
    
    #if args.gpu >= 0:
    chainer.cuda.get_device(0).use()  # Make a specified GPU current
    enc.to_gpu()  # Copy the model to the GPU
    dec.to_gpu()

    movie = MovieIter(INFILE)
    movie_out = cv2.VideoWriter(OUTFILE, cv2.VideoWriter_fourcc(*'DIB '), 24, (768,432))
    for frame in movie:
    
        inimg = loadimg(frame)
        #TODO: pic size should not be hard-coded
        #inary = np.zeros((3,768,1280), dtype=np.float32)
        #inary[:,0:720,0:1280] = inimg[:,0:720,0:1280]
        inary = np.zeros((3,512,768), dtype=np.float32)
        inary[:,0:432,0:768] = inimg[:,0:432,0:768]
        x_in = inary[np.newaxis,:]
        # x_in as an input image
        x_in = chainer.Variable(x_in)
        #if args.gpu >= 0:
        x_in.to_gpu()
        z = enc(x_in)
        x_out = dec(z)

        outimg = x_out.data.get()[0][:,0:432,0:768]
        del x_in
        del z
        del x_out
        bgrpic = to_bgr(outimg).copy()
        cv2.imshow(WINNAME, bgrpic)
        movie_out.write(bgrpic)
        key = cv2.waitKey(1)
        if key == ESC_KEY:
            break
            
    movie_out.release()
    
if __name__ == '__main__':
    # if (len(sys.argv) < 2):
        # print('requires 1 or more files')
        # quit()
        
    main()