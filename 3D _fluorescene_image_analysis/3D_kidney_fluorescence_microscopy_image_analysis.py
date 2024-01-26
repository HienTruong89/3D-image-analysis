import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy import ndimage as ndi
from skimage.data import kidney

#Import the 3d images 
import os 
from skimage import (exposure, util, feature, io, measure, morphology,
                     restoration, segmentation, transform)
image=kidney()

print('Type:',type(image))
print('Shape:',image.shape)
print('dtype:', image.dtype)
print('size:',image.size)

# Check the range of the image
vmin,vmax=image.min(),image.max()
vmean,vstd=image.mean(),image.std()
print(f'range: {vmin},{vmax}')
print(f'mean & std: {vmean},{vstd}')
# plot histogram
plt.hist(image.flatten())
plt.show()
# Plot the 2D image at half plane
fig,ax=plt.subplots(figsize=(10,8))
ax.imshow(image[image.shape[0]//2])

import plotly 
import plotly.express as px
img=px.imshow(image[image.shape[0]//2],zmax=vmax)
img.show()
# Checking image at different channels
fig = px.imshow(
    image[image.shape[0] // 2],
    facet_col=2,
    binary_string=True,
    labels={'facet_col': 'channel'}
)
plotly.io.show(fig)
# Checking range for each channel
vmin_0,vmin_1,vmin_2=image.min(axis=(0,1,2))
vmax_0,vmax_1,vmax_2=image.max(axis=(0,1,2))
print(f'range_channel_0: {vmin_0,vmax_0}')
print(f'range_channel_1: {vmin_1,vmax_1}')
print(f'range_channel_2: {vmin_2,vmax_2}')
# Animination of all planes 
fig_ani=px.imshow(image,
                  zmin=[vmin_0,vmin_1,vmin_2],
                  zmax=[vmax_0,vmax_1,vmax_2],
                  animation_frame=0,
                  binary_string=True,
                  labels={'animation_frame':'plane'})
fig_ani.show()
# Animination of all planes across channels
figx_ani=px.imshow(image,
                  zmin=[vmin_0,vmin_1,vmin_2],
                  zmax=[vmax_0,vmax_1,vmax_2],
                  animation_frame=0,
                  facet_col=3,
                  binary_string=True,
                  labels={'facet_col':'channel','animation_frame':'plane'})
figx_ani.show()

# Plot image using gray scale or certain range color 
red_kidney=image.copy()
redish=image[:,:,:,1]>200
red_kidney[redish]=[0,0,255]
plt.figure(figsize=(10,10))
plt.imshow(red_kidney[red_kidney.shape[0]//2])
from skimage import util
inverted_img=util.invert(red_kidney)
plt.figure(figsize=(10,10))
plt.imshow(inverted_img[inverted_img.shape[0]//2])
from skimage.color import rgb2gray 
gray_img=rgb2gray(image)
plt.figure(figsize=(10,10))
plt.imshow(gray_img[gray_img.shape[0]//2])
gray_img.shape
## Image processing
# Resize, Rescaling, Rotating and Flipping images 
from skimage import img_as_float,img_as_ubyte
from skimage.transform import resize,rescale
gray_transpose=gray_img.transpose(1,2,0)
gray_transpose.shape
resized_img=resize(gray_transpose,(480,480),anti_aliasing=True)
resized_img.shape
resized_img=resized_img.transpose(2,0,1)
resized_img.shape
plt.figure(figsize=(10,10))
plt.imshow(resized_img[resized_img.shape[0]//2])


rescaled_img=rescale(gray_img,0.4)
plt.figure(figsize=(10,10))
plt.imshow(rescaled_img[rescaled_img.shape[0]//2])
rescaled_img.shape
aspect_rescaled_ratio=rescaled_img.shape[1]/float(rescaled_img.shape[2])
aspect_rescaled_ratio

# drop image to the center --> recheck
def crop_center(image,crop_h,crop_w):
    height,width,planes=image.shape
    start_height=height//2-(crop_h//2)
    start_width=width//2-(crop_w//2)
    return image[start_height:start_height+crop_h,start_width:start_width+crop_w]
gray_crop=crop_center(gray_transpose,100,100)
gray_crop=gray_crop.transpose(2,0,1)
plt.figure(figsize=(10,10))
plt.imshow(gray_crop[gray_crop.shape[0]//2])
# denoise
from skimage.util import random_noise 
sigma=0.01
noise_img=random_noise(gray_img,var=sigma**2)
plt.figure(figsize=(10,10))
plt.imshow(noise_img[noise_img.shape[0]//2])

from skimage.restoration import denoise_tv_chambolle,denoise_bilateral 
from skimage.restoration import denoise_wavelet,estimate_sigma
sigma_est=estimate_sigma(noise_img,multichannel=True,average_sigmas=True)
sigma_est
# methods are better in order
denoise_1=denoise_tv_chambolle(noise_img,weight=0.007,multichannel=True)
plt.figure(figsize=(10,10))
plt.imshow(denoise_1[denoise_1.shape[0]//2])

denoise_3=denoise_wavelet(noise_img,multichannel=True)
plt.figure(figsize=(10,10))
plt.imshow(denoise_3[denoise_3.shape[0]//2])

denoise_2=denoise_bilateral(noise_img,sigma_color=0.05,sigma_spatial=2,multichannel=True)
plt.figure(figsize=(10,10))
plt.imshow(denoise_2[denoise_2.shape[0]//2])

## Normalisation to center pixel vales [-1,1] or [0,1]
X=gray_img.reshape(-1,512*512)
X.shape
plt.figure(figsize=(10,10))
plt.imshow(X[8].reshape(512,512,1))
plt.show() 
# [-1,1]
X.mean(axis=0).shape
mean=X.mean(axis=0) 
std=np.std(X, axis=0)
X_=(X-mean)/std

img_norm=X.reshape(-1,512,512)
plt.figure(figsize=(10,10))
plt.imshow(img_norm[8])
plt.show() 