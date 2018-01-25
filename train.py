#!/usr/bin/env python

import argparse, os
import numpy as np
from scipy import sparse, misc
from model.hashTable import hashTable
from tqdm import tqdm
from cgls import cgls
from utils import is_image_file, mod_crop

def main():
    parser = argparse.ArgumentParser(description="RAISR")
    parser.add_argument("--rate", type=int, default=3, help="upscale scale rate")
    parser.add_argument("--patch", type=int, default=11, help="image patch size")
    parser.add_argument("--Qangle", type=int, default=24, help="Training Qangle size")
    parser.add_argument("--Qstrength", type=int, default=3, help="Training Qstrength size")
    parser.add_argument("--Qcoherence", type=int, default=3, help="Training Qcoherence size")
    parser.add_argument('--datasets', default='./datasets/291/', type=str, help='path save the train dataset')

    opt = parser.parse_args()
    print(opt)

    rate = int(opt.rate)
    patch_size = int(opt.patch)

    images_path = [os.path.join(opt.datasets, x) for x in os.listdir(opt.datasets) if is_image_file(x)]
    print("Load dataset ", len(images_path))

    # Implement of Algorithm 1: Computing the hash-table keys.
    Qangle = opt.Qangle
    Qstrength = opt.Qstrength
    Qcoherence = opt.Qcoherence

    Q = np.zeros((Qangle, Qstrength, Qcoherence, rate*rate, patch_size*patch_size, patch_size*patch_size))
    V = np.zeros((Qangle, Qstrength, Qcoherence, rate*rate, patch_size*patch_size, 1))
    H = np.zeros((Qangle, Qstrength, Qcoherence, rate*rate, patch_size*patch_size))

    for image_path in tqdm(images_path):
        print("HashMap of %s" % image_path)
        im = misc.imread(image_path, mode='YCbCr')
        im_y = mod_crop(im[:, :, 0], rate)
        h, w = im_y.shape

        label = im_y
        data = misc.imresize(label, (h // rate, w // rate), interp='bicubic')

        for xP in range(0, data.shape[0] - data.shape[0]%patch_size, patch_size):
            for yP in range(0, data.shape[1] - data.shape[1]%patch_size, patch_size):

                im_patch = data[xP:xP+patch_size, yP:yP+patch_size]

                [angle, strength, coherence] = hashTable(im_patch, Qangle, Qstrength, Qcoherence)
                
                t = xP % rate * rate + yP % rate     # Pixel type

                X = label[xP, yP]       # GT pixle

                A = im_patch.reshape(1, -1)
            
                Q[angle,strength,coherence,t] += A * A.T
                V[angle,strength,coherence,t] += A.T * X

    
    operationcount = 0
    totaloperations = rate * rate * Qangle * Qstrength * Qcoherence
    for t in range(0, rate*rate):
        for angle in range(0, Qangle):
            for strength in range(0, Qstrength):
                for coherence in range(0, Qcoherence):
                    if round(operationcount*100/totaloperations) != round((operationcount+1)*100/totaloperations):
                        operationcount += 1
                        H[angle,strength,coherence,pixeltype] = cgls(Q[angle,strength,coherence,pixeltype], V[angle,strength,coherence,pixeltype])

    np.save("./filters", H)


if __name__ == '__main__':
    os.system('clear')
    main()
