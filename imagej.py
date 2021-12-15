# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 14:54:51 2021

@author: gniew
"""
import pandas as pd  
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_triangle, try_all_threshold
from skimage.morphology import remove_small_objects, remove_small_holes, skeletonize, closing, selem
from skimage.measure import label, regionprops
from skimage.segmentation import felzenszwalb
from skimage.color import label2rgb
from scipy import ndimage as ndi
import cv2
import tifffile as tif

def gray2color(u,channel):
    """
    Compute color image from intensity in fluorescence in a given channel.
    Arguments:
    -----------
        u: np.ndarray
            Input fluorescence image (2D).
        channel: int
            Channel to code the image in (0: Red, 1: Green, 2: Blue).
    Returns:
    -----------
        u_color: np.ndarray
            The computed output image in color.
    """
    u_color = np.dstack((
        rescale_intensity(u if channel==0 else np.zeros_like(u), out_range='float'),
        rescale_intensity(u if channel==1 else np.zeros_like(u), out_range='float'),
        rescale_intensity(u if channel==2 else np.zeros_like(u), out_range='float'),
        ))
    return u_color

fille_original = "C:\\Users\\gniew\\OneDrive\\Pulpit\\python\\moje\\rmtg\\RMTg_x5_12.tiff"
#fille_original = "C:\\Users\\malgo\\Desktop\\python\\rmtg\\RMTg_x5_12.tiff"
#fille_flipped = "C:\\Users\\malgo\\Desktop\\python\\rmtg\\test_photo.tif"
fille_flipped = "C:\\Users\\gniew\\OneDrive\\Pulpit\\python\\moje\\rmtg\\RMTg_x5_12_flipped.tif"

img_flipped = tif.imread(fille_flipped)
img_original = tif.imread(fille_original)


red_channel_flipped = img_flipped[:,:,0]
red_channel_original = img_original[:,:,0]

red_channel_rescale = rescale_intensity(red_channel_flipped, out_range='float')
red_channel_rescale1 = rescale_intensity(red_channel_flipped, out_range='float')

red_channel_rescale_original = rescale_intensity(red_channel_original, out_range='float')



# Chosen treshold method 
fig, ax = plt.subplots(2,2, sharex=False, sharey=False, figsize=(30,30))
ax[0,0].imshow(gray2color(red_channel_rescale,0))
ax[0,0].set_title('Red channel')

t = threshold_triangle(red_channel_rescale_original)

red_channel_rescale[red_channel_rescale<t] = 0
red_channel_rescale[red_channel_rescale>0] = 1
method_chosen = "threshold_triangle"
ax[0,1].imshow(red_channel_rescale, cmap ="gray")
ax[0,1].set_title(f'Threshold red channel, method: {method_chosen}')


ax[1,0].hist(red_channel_rescale1.flatten(), bins =100)
ax[1,0].set_title("Histogram from orginal data with treshold line")
ax[1,0].set_xticks(np.arange(0, 1.1, 0.1))
ax[1,0].axvline(t, color='r')

ax[1,1].hist(red_channel_rescale.flatten(), bins =10)
ax[1,1].set_title("Binarised data")


#segmentation with skimage and comercial filters
df = pd.DataFrame(columns= ["X", "Y", "X1", "Y1", "X2", "Y2"])
Red_mask = remove_small_objects(red_channel_rescale.astype(np.bool), min_size = 20)
Red_mask = remove_small_holes(Red_mask, area_threshold=2)
cells_a = []
label_image1 =label(Red_mask)
for i in regionprops(label_image1):
    if i.area < 1000:
        cells_a.append(i)

label_image2 = label_image1
label_image2[label_image2>0]
label_image2 = gray2color(label_image2,0)
for i in cells_a:
    dict_to_apend = {"X1": i.centroid[0], "Y1": i.centroid[1]}
    df = df.append(dict_to_apend, ignore_index=True)
fig, ax = plt.subplots(3,1, figsize = (50,50))
ax[0].set_aspect("equal")
ax[0].scatter(df["Y1"], df["X1"]*-1, color ="g", alpha = 0.7)
ax[0].set_title("Commercial methods")
ax[1].imshow(red_channel_rescale, cmap ="gray" )
y_min, y_max = ax[1].get_ylim()
x_min, x_max = ax[1].get_xlim()
ax[2].imshow(red_channel_rescale, cmap ="gray", extent = [x_min, x_max, y_min* -1, y_max* -1])
ax[2].scatter(df["Y1"], df["X1"]*-1, color ="r", alpha = 0.7, facecolors='none', s =50)
ax[2].set_xlim(x_min, x_max)
ax[2].set_ylim(y_min* -1, y_max* -1)