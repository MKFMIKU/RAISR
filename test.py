import numpy as np
from utils import is_image_file, mod_crop
from hashTable import hashTable


Qangle = 24
Qstrenth = 3
Qcoherence = 3
datasets = './datasets/General100/'
rate = 3

images_path = [os.path.join(datasets, x) for x in os.listdir(datasets) if is_image_file(x)]
print("Load dataset ", len(images_path))

H = np.load("filters.npy")

for image_path in tqdm(images_path):
    print("Test %s" % image_path)
    im = misc.imread(image_path, mode='YCbCr')
    lr = mod_crop(im, rate)
    y, cb, cr = lr.split()

    h, w = y.shape
    
    sr = np.zeros((h*rate, w*rate))

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