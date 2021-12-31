# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 11:56:09 2021

@author: malgo
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
import glob
import tifffile as tif
from skimage import measure, color, io

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

path_orginall = input("pls enter a path to batch_original folder: ")
path_orginall = path_orginall.replace("/", "//")
path_flipped =  input("pls enter a path to batch_flipped folder: ")
path_flipped = path_flipped.replace("/", "//")
save = input("please enter the path to the folder where the analysis will be saved: " )
save = path_flipped.replace("/", "//")
path_orginall_full = path_orginall + "\\*.*"
path_flipped_full = path_flipped + "\\*.*"

orginal_list = glob.glob(path_orginall_full)
flipped_list = glob.glob(path_flipped_full)





def eve(o, f, save = False):
    df_watershed = pd.DataFrame(columns= ["X", "Y", "rat_id", "AP"])
    df_comercial_labeling = pd.DataFrame(columns= ["X", "Y", "rat_id", "AP"])
    result_watershed = pd.DataFrame(columns = ["cells", "rat_id", "AP"])
    result_comercial_labeling = pd.DataFrame(columns = ["cells", "rat_id", "AP"])
    for i in range(len(o)):
        czarek = o[i].split("\\")
        last_word = czarek[-1].split("_")
        img_flipped = cv2.imread(f[i])
        img_original = cv2.imread(o[i])
        red_channel_flipped = img_flipped[:,:,2]
        red_channel_original = img_original[:,:,2]

        red_channel_rescale = rescale_intensity(red_channel_flipped, out_range='float')
        red_channel_rescale1 = rescale_intensity(red_channel_flipped, out_range='float')

        red_channel_rescale_original = rescale_intensity(red_channel_original, out_range='float')
        # Chosen treshold method 
        t = threshold_triangle(red_channel_rescale_original)
        
        red_channel_rescale[red_channel_rescale<t] = 0
        red_channel_rescale[red_channel_rescale>0] = 1

#segmentation with skimage and comercial filters
        Red_mask = remove_small_objects(red_channel_rescale.astype(np.bool), min_size = 20)
        Red_mask = remove_small_holes(Red_mask, area_threshold=2)
        cells_a = []
        label_image1 =label(Red_mask)
        for j in regionprops(label_image1):
            if j.area < 1000:
                cells_a.append(j)
        
        for k in cells_a:
            dict_to_apend = {"X": k.centroid[0], "Y": k.centroid[1]}
            df_comercial_labeling = df_comercial_labeling.append(dict_to_apend, ignore_index=True)
        fig, ax = plt.subplots(3,1, figsize = (50,50))
        ax[0].set_aspect("equal")
        ax[0].scatter(df_comercial_labeling["Y"], df_comercial_labeling["X"]*-1, color ="g", alpha = 0.7)
        ax[0].set_title(f"Commercial methods, {last_word[0]} AP: {last_word[1]}")
        ax[1].imshow(red_channel_rescale, cmap ="gray" )
        y_min, y_max = ax[1].get_ylim()
        x_min, x_max = ax[1].get_xlim()
        ax[2].imshow(red_channel_rescale, cmap ="gray", extent = [x_min, x_max, y_min* -1, y_max* -1])
        ax[2].scatter(df_comercial_labeling["Y"], df_comercial_labeling["X"]*-1, color ="r", alpha = 0.7, facecolors='none', s =50)
        ax[2].set_xlim(x_min, x_max)
        ax[2].set_ylim(y_min* -1, y_max* -1)
       
        # watershed approach
        
        kernel = np.ones((2,2), np.uint8)
        red_mask_int = Red_mask.astype(np.float64)
        openning = cv2.morphologyEx(red_mask_int, cv2.MORPH_OPEN, kernel, iterations = 1)
        #openning = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel, iterations = 1)
        surebg = cv2.dilate(openning, kernel, iterations = 5)
        dist_transform = cv2.distanceTransform(openning.astype(np.uint8), cv2.DIST_L2, 3) # int = mask Size 
        ret, sure_fg = cv2.threshold(dist_transform, 0.45*dist_transform.max(), 255, 0) # the mbigger the  treshold the more watersheded are cells but small cell are lost
        sure_fg = sure_fg.astype(np.uint8)
        unknown = cv2.subtract(surebg, sure_fg.astype(np.float64))

        ret1, markers = cv2.connectedComponents(sure_fg)
        markers = markers+10
        markers[unknown == 1] = 0
        markers = markers.astype(np.int32)
        markers = cv2.watershed(img_flipped, markers)
        img_flipped[markers == -1] = [0, 255, 255]
        img2 = color.label2rgb(markers, bg_label = 0)
        #reszie = cv2.resize(surebg, (1000, 1000))
        #reszie1 = cv2.resize(openning, (1000, 1000))
        #resize2 = cv2.resize(dist_transform, (1000, 1000))
        #resize3 = cv2.resize(sure_fg, (1000, 1000))
        #resize4 = cv2.resize(unknown, (1000, 1000))
        #resize5 = cv2.resize(img2, (1000, 1000))
        #cv2.imshow("surebg", reszie)
        #cv2.imshow("opening", reszie1)
        #cv2.imshow("Sure_FG", resize3)
        #cv2.imshow("Unknow", resize4)
        #cv2.imshow("watershed", resize5)
        #cv2.imshow("kupa", resize2)
        #cv2.waitKey(0)
        cells_watershed = []
        for l in regionprops(markers):
            if l.area < 500:
                cells_watershed.append(l)
        
        for m in cells_watershed:
            dict_to_apend = {"X": m.centroid[0], "Y": m.centroid[1], "rat_id": last_word[0], "AP": last_word[1]}
            df_watershed = df_watershed.append(dict_to_apend, ignore_index=True)
        dict2_to_apend = {"cells": len(cells_watershed), "rat_id": last_word[0], "AP": last_word[1]}
        result_watershed_watershed = result_watershed_watershed.append(dict2_to_apend, ignore_index=True)
        fig1, ax = plt.subplots(3,1, figsize = (50,50))
        ax[0].set_aspect("equal")
        ax[0].scatter(df_watershed["Y"], df_watershed["X"]*-1, color ="g", alpha = 0.7)
        ax[0].set_title("Watershed")
        ax[1].imshow(red_channel_rescale, cmap ="gray" )
        y_min, y_max = ax[1].get_ylim()
        x_min, x_max = ax[1].get_xlim()
        ax[2].imshow(red_channel_rescale, cmap ="gray", extent = [x_min, x_max, y_min* -1, y_max* -1])
        ax[2].scatter(df_watershed["Y"], df_watershed["X"]*-1, color ="y", alpha = 0.7, facecolors='none', s =50)
        ax[2].set_xlim(x_min, x_max)
        ax[2].set_ylim(y_min* -1, y_max* -1)
        
    return df_watershed, result_watershed_watershed

water, result_watershed_watershed = eve(orginal_list, flipped_list)