# -*- coding: utf-8 -*-

import numpy as np

def hashTable(patch,Qangle,Qstrenth,Qcoherence):
    [gx,gy] = np.gradient(patch)
    G = np.matrix((gx.ravel(),gy.ravel())).T
    x = G.T*G
    [eigenvalues,eigenvectors] = np.linalg.eig(x)
    
    #For angle
    angle = np.math.atan2(eigenvectors[0,1],eigenvectors[0,0])
    if angle<0:
        angle += np.pi
    
    #For strength
    strength = eigenvalues.max()/eigenvalues.sum()
    
    #For coherence
    lamda1 = np.math.sqrt(eigenvalues.max())
    lamda2 = np.math.sqrt(eigenvalues.min())
    coherence = np.abs((lamda1-lamda2)/(lamda1+lamda2))
    
    #Quantization
    angle = np.floor(angle/(np.pi/Qangle)-1)
    strength = np.floor(strength/(1/Qstrenth)-1)
    coherence = np.floor(coherence/(1/Qcoherence)-1)
    
    return angle,strength,coherence