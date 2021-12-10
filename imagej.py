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
#from gray2color import gray2color
print ("Cycki są fajniejsze")
print ("ale i tyłeczki rządzą")
print("jak kto woli")
print("Kamil to covidowiec")
print(" noga noga")
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


#fille = "C:\\Users\\malgo\\Desktop\\python\\rmtg\\test_photo.tif"
fille = "C:\\Users\\gniew\\OneDrive\\Pulpit\\python\\moje\\rmtg\\test_photo.tif"
img = tif.imread(fille)
red_channel = img[:,:,0]
one_d_red_channel = red_channel.flatten()
red_channel_rescale = rescale_intensity(red_channel, out_range='float')
red_channel_rescale1 = rescale_intensity(red_channel, out_range='float')
# try diffrent method 
fig, ax = try_all_threshold(red_channel_rescale, figsize=(30,30), verbose=True)

# Chosen treshold method 
fig, ax = plt.subplots(2,2, sharex=False, sharey=False, figsize=(30,30))
ax[0,0].imshow(gray2color(red_channel_rescale,0))
ax[0,0].set_title('Red channel')
t = threshold_triangle(red_channel_rescale)
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
#segmenation with skimage 
label_image = label(red_channel_rescale)
cells_filter = []
cells = []
for i in regionprops(label_image):
    if i.area < 1000:
        cells.append(i.area)
plt.hist(cells, bins= 80)
for i in regionprops(label_image):
    if 4 < i.area < 1000:
        cells_filter.append(i)
        


df = pd.DataFrame(columns= ["X", "Y", "X1", "Y1", "X2", "Y2"])
for i in cells_filter:
    dict_to_apend = {"X": i.centroid[0], "Y": i.centroid[1]}
    df = df.append(dict_to_apend, ignore_index=True)
fig, ax = plt.subplots(1,2, figsize = (50,50))
ax[0].set_aspect("equal")
ax[0].scatter(df["Y"], df["X"]*-1, color ="r", alpha = 0.7)
ax[0].set_title("My methods")
ax[1].imshow(red_channel_rescale, cmap ="gray")
ax[1].set_title("Binarised data")
#segmentation with skimage and comercial filters
Red_mask = remove_small_objects(red_channel_rescale.astype(np.bool), min_size = 4)
Red_mask = remove_small_holes(Red_mask, area_threshold=2)
cells_a = []
label_image1 =label(Red_mask)
for i in regionprops(label_image1):
    if i.area < 1000:
        cells_a.append(i)

for i in cells_a:
    dict_to_apend = {"X1": i.centroid[0], "Y1": i.centroid[1]}
    df = df.append(dict_to_apend, ignore_index=True)
fig, ax = plt.subplots(1,2, figsize = (50,50))
ax[0].set_aspect("equal")
ax[0].scatter(df["Y1"], df["X1"]*-1, color ="g", alpha = 0.7)
ax[0].set_title("Commercial methods")
ax[1].imshow(red_channel_rescale, cmap ="gray")
#segmentation with felzenszwalb and comercial filters
cells_b = []
label_image2 = felzenszwalb(Red_mask, scale = 2)
for i in regionprops(label_image2):
    if i.area < 1000:
        cells_b.append(i)

for i in cells_b:
    dict_to_apend = {"X2": i.centroid[0], "Y2": i.centroid[1]}
    df = df.append(dict_to_apend, ignore_index=True)
    
fig, ax = plt.subplots(1,3, figsize = (50,50))
ax[0].set_aspect("equal")
ax[0].scatter(df["Y2"], df["X2"]*-1, color ="m", alpha = 0.7, s =10, facecolors='none')
ax[0].set_title("Felzenszwalb method")
x_range = ax[0].get_xlim()
y_range = ax[0].get_ylim()

ax[1].imshow(red_channel_rescale, cmap ="gray")
ax[1].set_title("Binarised data")
x_range1 = ax[1].get_xlim()
y_range1 = ax[1].get_ylim()
mean_y = mean(y_range1)
d = 12
ax[2].imshow(red_channel_rescale, cmap ="gray", extent =[x_range[0], x_range[1], y_range[0], y_range[1]])

ax[2].scatter(df["Y2"], df["X2"]*-1, color ="m", alpha = 0.7, s =10, facecolors='none')
