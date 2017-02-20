# -*- coding: utf-8 -*-

import os  
import os.path
import numpy as np
import cv2
from sklearn import linear_model 

from hashTable import hashTable

# The const variable
patchSize = 10
R = 2
Qangle = 24
Qstrenth = 3
Qcoherence = 3

# Begin
dataDir="../train"
fileList = []
for parent,dirnames,filenames in os.walk(dataDir):
    for filename in filenames:    
        fileList.append(os.path.join(parent,filename)) 

# Init the Q and V        
Q = np.zeros((Qangle*Qstrenth*Qcoherence,R*R,patchSize*patchSize,patchSize*patchSize))
V = np.zeros((Qangle*Qstrenth*Qcoherence,R*R,patchSize*patchSize))

for file in fileList:
    mat = cv2.imread(file,0)
    HMat,WMat = mat.shape[:2]
    
    # Use upsampling to get the LRimage
    # which the HRImage is the origin
    LRImage =cv2.resize(mat,(0,0),fx=0.5,fy=0.5)
    HRImage = mat

    # Upscale the LRImage
    LRImage = cv2.resize(LRImage,(WMat,HMat),cv2.INTER_LINEAR)

    # for each pixel k in yi do
    iPixel = 5
    while iPixel<HMat-5:
        jPixel = 5
        while jPixel<WMat-5:
            patch = LRImage[iPixel-5:iPixel+5,jPixel-5:jPixel+5]
            [angle,strength,coherence] = hashTable(patch,Qangle,Qstrenth,Qcoherence)
            j = angle+strength+coherence
            patch = patch.reshape(1,-1)
            x = HRImage[iPixel,jPixel]
            t = iPixel%R*R+jPixel%R
            Q[j,t] = Q[j,t]+patch.T*patch
            V[j,t] = V[j,t]+patch*x
            jPixel = jPixel+1
        iPixel = iPixel+1


H = np.zeros((Qangle*Qstrenth*Qcoherence,R*R,patchSize*patchSize))    
# For each key j and pixel-type t        
for t in range(R*R):
    for j in range(Qangle*Qstrenth*Qcoherence):
        reg = linear_model.LinearRegression()
        reg.fit(Q[j,t],V[j,t])
        H[j,t] = reg.coef_

print("Train is off\nTest is begin \n")

iPixel = 5
while iPixel<HMat-5:
    jPixel = 5
    while jPixel<WMat-5:
        patch = LRImage[iPixel-5:iPixel+5,jPixel-5:jPixel+5]
        [angle,strength,coherence] = hashTable(patch,Qangle,Qstrenth,Qcoherence)
        j = angle+strength+coherence
        t = iPixel%R*R+jPixel%R
        h = H[j,t]
        patch = patch.reshape(1,-1)
        LRImage[iPixel,jPixel] = patch*np.matrix(h).T   
        jPixel = jPixel+1
    iPixel = iPixel+1
    
cv2.imwrite("Result.bmp",LRImage)