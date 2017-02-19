# -*- coding: utf-8 -*-

import numpy as np

def hashTable(patch,Qangle,Qstrenth,Qcoherence):
   
    # Composed G from the horizontal and vertical gradients,
    # gx and gy, of the surroundings of the k-th pixel
    [gx,gy] = np.gradient(patch)
    gx = gx.ravel()
    gy = gy.ravel()
    G = np.matrix((gx,gy))

    # Use eigen-decomposition 
    x = G*G.T
    
    # Compuetr the angle
    [eigenvalues,eigenvectors] = np.linalg.eig(x)
    angle = np.math.atan2(eigenvectors[0,1],eigenvectors[0,0])
    if angle<0:
        angle += np.pi
    angle=np.ceil(angle/Qangle)
          
    # Computer the strength
    stength = np.ceil((eigenvalues.max()/Qstrenth))
    
    # Computer the coherence
    lamda1 = np.math.sqrt(eigenvalues.max())
    lamda2 = np.math.sqrt(eigenvalues[1] if eigenvalues[0] == eigenvalues.max() else eigenvalues[0])
    coherence = np.abs((lamda1-lamda2)/(lamda1+lamda2))
    coherence = (coherence/Qcoherence)
    
    return angle,stength,coherence