import numpy as np
import cv2
import matplotlib.pyplot as plt

from hashTable import hashTable


Qangle = 24
Qstrenth = 3
Qcoherence = 3

mat = cv2.imread("../train/a.jpg")
h = np.load("Filters.npy")

mat = cv2.cvtColor(mat, cv2.COLOR_BGR2YCrCb)[:,:,2]
# LR = cv2.resize(LuminanceMat,(0,0),fx=0.5,fy=0.5)
# LR = cv2.GaussianBlur(LR,(0,0),2)

# Upscaling
LR = cv2.resize(mat,(0,0),fx=2,fy=2)

LRDirect = np.zeros((LR.shape[0],LR.shape[1]))
for xP in range(5,LR.shape[0]-6):
    for yP in range(5,LR.shape[1]-6):
        patch = LR[xP-5:xP+6,yP-5:yP+6]
        [angle,strenth,coherence] = hashTable(patch,Qangle,Qstrenth,Qcoherence)
        j = angle*9+strenth*3+coherence
        A = patch.reshape(1,-1)
        t = xP%2*2+yP%2
        hh = np.matrix(h[j,t])
        LRDirect[xP][yP] = hh*A.T
        
        
print("Test is off")
        
# Show the result
mat = cv2.imread("../train/a.jpg")
mat = cv2.cvtColor(mat, cv2.COLOR_BGR2YCrCb)

fig, axes = plt.subplots(ncols=2,figsize=(15,10))
axes[0].imshow(cv2.cvtColor(mat, cv2.COLOR_YCrCb2RGB))
axes[0].set_title('ORIGIN')


LR = cv2.resize(mat,(0,0),fx=2,fy=2)
LRDirectImage = LR
LRDirectImage[:,:,2] = LRDirect
axes[1].imshow(cv2.cvtColor(LRDirectImage, cv2.COLOR_YCrCb2RGB))
axes[1].set_title('RAISR')

fig.savefig("../fig.png")