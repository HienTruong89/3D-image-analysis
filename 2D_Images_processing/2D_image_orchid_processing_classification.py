import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from skimage import io
from skimage.color import rgb2gray
os.chdir('C:/Users\Hien Thi Dieu Truong/Documents/Python modelling/Git hub/3D-image-analysis/2D_Images_processing')
img1=io.imread('orchid_1.jpg')
img1.shape
img2=io.imread('orchid_2.jpg')
img3=img1.copy()
img4=img2.copy()
plt.figure(figsize=(10,10))
plt.imshow(img1,cmap='gray')

# create noise image
from skimage.util import random_noise
sigma=0.02
img5=random_noise(img1,var=sigma**2)
img6=random_noise(img2,var=sigma**2)
plt.figure(figsize=(10,10))
plt.imshow(img5,cmap='gray')
# crop image to the center
def crop_image(image,crop_h,crop_w):
    height,width,channels=image.shape
    start_height=height//2 - crop_h//2
    start_width=width//2 - crop_w//2
    
    return image[start_height:start_height+crop_h,start_width:start_width+crop_w]

img7=img6.copy()
img8=crop_image(img7,1000,1000)
img9=crop_image(img5,1000,1000)
plt.figure(figsize=(10,10))
plt.imshow(img9,cmap='gray')

# Change to gray scale
img1=rgb2gray(img1)
img2=rgb2gray(img2)
img3=rgb2gray(img3)
img4=rgb2gray(img4)
img5=rgb2gray(img5)
img6=rgb2gray(img6)
img7=rgb2gray(img7)
img8=rgb2gray(img8)
img9=rgb2gray(img9)
# Resize image 
from skimage.transform import resize
img1_resized=resize(img1,output_shape=(100,100),mode='reflect',anti_aliasing=False)
img2_resized=resize(img2,output_shape=(100,100),mode='reflect',anti_aliasing=False)
img3_resized=resize(img3,output_shape=(100,100),mode='reflect',anti_aliasing=False)
img4_resized=resize(img4,output_shape=(100,100),mode='reflect',anti_aliasing=False)
img5_resized=resize(img5,output_shape=(100,100),mode='reflect',anti_aliasing=False)
img6_resized=resize(img6,output_shape=(100,100),mode='reflect',anti_aliasing=False)
img7_resized=resize(img7,output_shape=(100,100),mode='reflect',anti_aliasing=False)
img8_resized=resize(img8,output_shape=(100,100),mode='reflect',anti_aliasing=False)
img9_resized=resize(img9,output_shape=(100,100),mode='reflect',anti_aliasing=False)
plt.figure(figsize=(10,10))
plt.imshow(img8_resized,cmap='gray')
# Combine the image into one array
train=np.concatenate((img1_resized,img2_resized,img3_resized,img4_resized,img5_resized,img6_resized,img7_resized),axis=0)
test=np.concatenate((img8_resized,img9_resized),axis=0)

# Prepare the data into 1 axis
train=train.reshape(-1,10000).astype(np.float32)
test=test.reshape(-1,10000).astype(np.float32)

train.shape
test.shape
# Create labels for train and test data. 
train_labels={'img1':'White Phalaenopsis','img2':'Purple Phalaenopsis','img3':'White Phalaenopsis',
              'img4':'Purple Phalaenopsis','img5':'White Phalaenopsis',
              'img6':'Purple Phalaenopsis','img7':'Purple Phalaenopsis'}
test_labels={'img8':'Purple Phalaenopsis','img9':'White Phalaenopsis'}

code=np.array([0,1,0,1,0,1,1])
tr_labels=pd.DataFrame(list(train_labels.items()),columns=['Image','Labels'])
tr_labels['code']=code
test_labels=pd.DataFrame(list(test_labels.items()),columns=['Image','Labels'])
test_labels['code']=np.array([1,0])

# Initiate kNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
model=knn.fit(train,tr_labels['code'])
results=model.predict(test)

# Calculate the classification results
from sklearn.metrics import accuracy_score, recall_score,precision_score
print('accuracy_score:', accuracy_score(test_labels['code'],results))
print('recall_score:',recall_score(test_labels['code'],results))
print('precision_score:',precision_score(test_labels['code'],results))

# Print the name of results
for i in results:
    if i==0:
        print('White Phalaenopsis')
    else:
        print('Purple Phalaenopsis')