# -*- coding: utf-8 -*-

import os  
import numpy as np
import cv2

from hashTable import hashTable

from sklearn import linear_model 

# The const variable
patchSize = 11
R = 2
Qangle = 24
Qstrenth = 4
Qcoherence = 4

# Begin
dataDir="../train"
fileList = []
for parent,dirnames,filenames in os.walk(dataDir):
    for filename in filenames:    
        fileList.append(os.path.join(parent,filename))

mat = cv2.imread(fileList[0],0)
LRImage =cv2.resize(mat,(0,0),fx=0.5,fy=0.5)
HRImage = mat

HMat,WMat = mat.shape[:2]

LRImage = cv2.resize(LRImage,(WMat,HMat),2,2,cv2.INTER_LINEAR)

patch = LRImage[0:patchSize,0:patchSize]

[angle,strength,coherence] = hashTable(patch,Qangle,Qstrenth,Qcoherence)

# Init the Q and V        
Q = np.zeros((Qangle,Qstrenth,Qcoherence,patchSize*patchSize,patchSize*patchSize))
V = np.zeros((Qangle,Qstrenth,Qcoherence,patchSize*patchSize))

for file in fileList:
    mat = cv2.imread(fileList[0],0)
    LRImage =cv2.resize(mat,(0,0),fx=0.5,fy=0.5)
    HRImage = mat
    HMat,WMat = mat.shape[:2] 
    LRImage = cv2.resize(LRImage,(WMat,HMat),cv2.INTER_LINEAR)
    
    iPixel = 6
    while iPixel<HMat-5:
        jPixel = 6
        while jPixel<WMat-5:
            patch = LRImage[iPixel-6:iPixel+5,jPixel-6:jPixel+5]
            [angle,strength,coherence] = hashTable(patch,Qangle,Qstrenth,Qcoherence)
            patch = patch.reshape(1,-1)
            x = HRImage[iPixel,jPixel]
            Q[angle,strength,coherence] = Q[angle,strength,coherence] + patch.T*patch
            V[angle,strength,coherence] = V[angle,strength,coherence] + patch*x
            jPixel = jPixel+1
        iPixel = iPixel+1    
    
H = np.zeros((Qangle,Qstrenth,Qcoherence,patchSize*patchSize))    
# For each key j and pixel-type t        
for i1 in range(Qangle):
    for i2 in range(Qstrenth):
        for i3 in range(Qcoherence):
            reg = linear_model.LinearRegression()
            reg.fit(Q[i1,i2,i3],V[i1,i2,i3])
            H[i1,i2,i3] = reg.coef_

print("Train is off\n")        
np.save("HashTable",H)

testMat = cv2.imread(fileList[1],0)
cv2.imwrite("Origin.bmp",testMat)
LRImage =cv2.resize(testMat,(0,0),fx=0.5,fy=0.5)
cv2.imwrite("LRImage.bmp",LRImage)

LRImage = cv2.resize(LRImage,(WMat,HMat),cv2.INTER_LINEAR)

while iPixel<HMat-5:
    jPixel = 6
    while jPixel<WMat-5:
        patch = LRImage[iPixel-6:iPixel+5,jPixel-6:jPixel+5]
        [angle,strength,coherence] = hashTable(patch,Qangle,Qstrenth,Qcoherence)
        f = H[angle,strength,coherence]
        patch = patch.reshape(1,-1)
        LRImage[jPixel,iPixel] = patch*np.matrix(f).T
        jPixel = jPixel+1
    iPixel = iPixel+1    
cv2.imwrite("LRImage_RAISR.bmp",LRImage)

'''
for file in fileList:
    mat = cv2.imread(file,0)
    
    # Use upsampling to get the LRimage
    # which the HRImage is the origin
    LRImage = cv2.GaussianBlur(mat,(5,5),0)
    HRImage = mat
    HMat,WMat = mat.shape[:2]    

    # Upscale the LRImage
    LRImage = cv2.resize(LRImage,(WMat,HMat),cv2.INTER_LINEAR)

    # for each pixel k in yi do
    iPixel = 6
    while iPixel<HMat-5:
        jPixel = 6
        while jPixel<WMat-5:
            patch = LRImage[iPixel-6:iPixel+5,jPixel-6:jPixel+5]
            try:
                j = [angle,strength,coherence] = hashTable(patch,Qangle,Qstrenth,Qcoherence)
            except:
                print("X Y",iPixel,jPixel)
            patch = patch.reshape(1,-1)
            x = HRImage[iPixel,jPixel]
            Q[angle,strength,coherence] = Q[angle,strength,coherence] + patch.T*patch
            V[angle,strength,coherence] = V[angle,strength,coherence] + patch*x
            
            jPixel = jPixel+1
        iPixel = iPixel+1


H = np.zeros((Qangle,Qstrenth,Qcoherence,patchSize*patchSize))    
# For each key j and pixel-type t        
for t in range(R*R):
    for j in range(Qangle*Qstrenth*Qcoherence):
        reg = linear_model.LinearRegression()
        reg.fit(Q[j,t],V[j,t])
        H[j,t] = reg.coef_

np.save("HashTable",H)

print("Train is off\n")
'''