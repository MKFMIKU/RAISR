#!/usr/bin/env python

import argparse, os
import numpy as np
from scipy import sparse, misc
from model.hashTable import hashTable
from tqdm import tqdm


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', 'bmp'])


def mod_crop(im, modulo):
    sz = im.shape
    h = sz[0] // modulo * modulo
    w = sz[1] // modulo * modulo
    ims = im[0:h, 0:w, ...]
    return ims


def main():
    parser = argparse.ArgumentParser(description="RAISR")
    parser.add_argument("--scale", type=int, default=3, help="Training Qangle size")
    parser.add_argument("--patch", type=int, default=11, help="Training Qangle size")
    parser.add_argument("--Qangle", type=int, default=24, help="Training Qangle size")
    parser.add_argument("--Qstrenth", type=int, default=3, help="Training Qstrenth size")
    parser.add_argument("--Qcoherence", type=int, default=3, help="Training Qcoherence size")
    parser.add_argument('--datasets', default='./datasets/BSDS200/', type=str, help='path save the train dataset')

    opt = parser.parse_args()
    print(opt)

    scale = int(opt.scale)
    patch = int(opt.patch)

    images_path = [os.path.join(opt.datasets, x) for x in os.listdir(opt.datasets) if is_image_file(x)]
    print("Load dataset ", len(images_path))

    # Implement of Algorithm 1: Computing the hash-table keys.
    Qangle = opt.Qangle
    Qstrenth = opt.Qstrenth
    Qcoherence = opt.Qcoherence

    Q = np.zeros((Qangle * Qstrenth * Qcoherence, 4, patch * patch, patch * patch))
    V = np.zeros((Qangle * Qstrenth * Qcoherence, 4, patch * patch, 1))
    res_h = np.zeros((Qangle * Qstrenth * Qcoherence, 4, patch * patch))

    for image_path in tqdm(images_path):
        print("HashMap of %s" % image_path)
        im = misc.imread(image_path, mode='YCbCr')
        im_y = mod_crop(im[:, :, 0], scale)
        h, w = im_y.shape

        label = im_y
        data = misc.imresize(label, (h // scale, w // scale), interp='bicubic')

        # Todo: Add more elegant to crop path
        for xP in range(patch, data.shape[0], patch):
            for yP in range(patch, data.shape[1], patch):
                im_patch = data[xP:xP + patch, yP:yP + patch]
                [angle, strenth, coherence] = hashTable(im_patch, Qangle, Qstrenth, Qcoherence)
                j = angle * 9 + strenth * 3 + coherence
                A = im_patch.reshape(1, -1)
                x = label[xP][yP]

                t = xP % 2 * 2 + yP % 2
                Q[j, t] += A * A.T
                V[j, t] += A.T * x

    # Todo: Change to more quick learning style
    for t in range(4):
        for j in range(Qangle * Qstrenth * Qcoherence):
            res_h[j, t] = sparse.linalg.cg(Q[j, t], V[j, t])[0]

    np.save("./Filters", res_h)


if __name__ == '__main__':
    os.system('clear')
    main()
