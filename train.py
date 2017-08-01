import os
import numpy as np
import cv2
from scipy import sparse

from model.hashTable import hashTable


Qangle = 24
Qstrenth = 3
Qcoherence = 3

Q = np.zeros((Qangle*Qstrenth*Qcoherence,4,11*11,11*11))
V = np.zeros((Qangle*Qstrenth*Qcoherence,4,11*11,1))
h = np.zeros((Qangle*Qstrenth*Qcoherence,4,11*11))


dataDir="../train"
fileList = []
for parent,dirnames,filenames in os.walk(dataDir):
    for filename in filenames:
        fileList.append(os.path.join(parent,filename))

for file in fileList:
    print("HashMap of %s"%file)
    mat = cv2.imread(file)
    #转换色彩空间，只对亮度进行训练
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2YCrCb)[:,:,2]
    #放缩为0-1
    mat = cv2.normalize(mat.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    HR = mat

    #模糊LR
    #Upscaling
    LR = cv2.GaussianBlur(LR,(0,0),2);

    # Set the train map

    #遍历每个像素
    for xP in range(5,LR.shape[0]-6):
        for yP in range(5,LR.shape[1]-6):
        	#取patch
            patch = LR[xP-5:xP+6,yP-5:yP+6]
            #计算特征
            [angle,strenth,coherence] = hashTable(patch,Qangle,Qstrenth,Qcoherence)
            #压缩向量空间
            j = angle*9+strenth*3+coherence
            A = patch.reshape(1,-1)
            x = HR[xP][yP]
            #计算像素类型
            t = xP%2*2+yP%2
            #存入对应的HashMap
            Q[j,t] += A*A.T
            V[j,t] += A.T*x


# Set the train step
for t in range(4):
    for j in range(Qangle*Qstrenth*Qcoherence):
    	#针对每种像素类型和图像特征训练4*24*3*3个过滤器
        h[j,t] = sparse.linalg.cg(Q[j,t],V[j,t])[0]

print("Train is off")
np.save("./Filters",h)
