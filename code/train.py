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

# Code single test        
mat = cv2.imread(fileList[0],0)
mat = cv2.normalize(mat.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
H,W = mat.shape[:2]

# Use upsampling to get the LRimage
# which the HRImage is the origin
LRImage =cv2.resize(mat,(0,0),fx=0.5,fy=0.5)
HRImage = mat

# Upscale the LRImage
LRImage = cv2.resize(LRImage,(W,H),cv2.INTER_LINEAR)

Q = np.zeros((Qangle*Qstrenth*Qcoherence,R*R,patchSize*patchSize,patchSize*patchSize))
V = np.zeros((Qangle*Qstrenth*Qcoherence,R*R,patchSize*patchSize))
# for each pixel k in yi do
i1 = 0
while i1+patchSize<H:
    j1 = 0
    while j1+patchSize<W:
        patch = LRImage[i1:i1+patchSize,j1:j1+patchSize]
        j = [angle,stength,coherence] = hashTable(patch,Qangle,Qstrenth,Qcoherence)
        patch = patch.reshape(1,-1)
        x = HRImage[i1,j1]
        t = i1%R*R+j1%R+1
        A = patch.T*patch
        b = patch.T*x
        #Q[j,t] = Q[j,t]+A
        #V[j,t] = V[j,t]+b
        #for each
        j1 = j1+patchSize
    i1 = i1+patchSize

# for each key j and pixel-type t
#for r in R*R:
#    for j in Qangle*Qstrenth*Qcoherence:
        
                
#np.concatenate((Q,A),axis=0)

# Use eigen-decomposition to computer angel

# Use scikit-learn library to get least-squares
# dataset_Q = []
# dataset_V = []

'''
A = np.empty(((H-10)*(W-10),100))
for i1 in range(H-10):
    for j1 in range(W-10):
        patchL = LRImage[i1:i1+10,j1:j1+10]
        patchL = patchL.ravel().T                     
        A[i1*j1]=patchL
Q = A.T*A
V = A.T*HRImage    
'''
     
# A = np.matrix(A)        
# Sampling K patches/pixels from the images

'''
b = HRImage[0:10,0:10]
Q = A.T*A
'''

# dataset_Q.append(Q)
# dataset_V.append(V)    
        
# dataset_Q = np.array(dataset_Q)
# regr = linear_model.LinearRegression()
# regr.fit(Q, V)
