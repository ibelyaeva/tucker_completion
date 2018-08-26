import nilearn


from nilearn import image
import nibabel as nib
import copy
from nilearn import plotting
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from math import ceil
from nilearn.datasets import MNI152_FILE_PATH
from sklearn.model_selection import train_test_split
from nibabel.affines import apply_affine
from nilearn.image.resampling import coord_transform, get_bounds, get_mask_bounds
from nilearn.image import resample_img
from nilearn.masking import compute_background_mask
import matplotlib.gridspec as gridspec
import matplotlib
from nilearn.plotting import find_xyz_cut_coords

import metric_util as mc
import math
import math_format as mf

PROJECT_DIR  = "/work/pl/sch/analysis/scripts"
PROJECT_ROOT_DIR = "."
DATA_DIR = "data"
CSV_DATA = "csv_data"
FIGURES = "figures"

#SMALL_SIZE = 8
#matplotlib.rc('font', size=SMALL_SIZE)
#matplotlib.rc('axes', titlesize=SMALL_SIZE)
#matplotlib.rc('figure', titlesize=SMALL_SIZE)

       
def save_report_fig(fig_id, tight_layout=True):
        path = os.path.join(PROJECT_DIR, FIGURES, fig_id + ".png")
        print("Saving figure", path)
        plt.savefig(path, format='png', dpi=300)

def save_fig_abs_path(fig_id, tight_layout=True):
    path = os.path.join(fig_id + ".png")
    print("Saving figure", path)
    plt.savefig(path, format='png', facecolor='k', edgecolor='k', dpi=300)
        
def save_csv(df, dataset_id):
    path = os.path.join(PROJECT_DIR, CSV_DATA, dataset_id + ".csv")
    print("Saving dataset", path)
    df.to_csv(path)

def save_csv_by_path(df, file_path, dataset_id):
    path = os.path.join(file_path, dataset_id + ".csv")
    print("Saving dataset", path)
    df.to_csv(path)
    
def save_fig_png(fig_id, tight_layout=True):
        path = os.path.join(fig_id + ".png")
        print("Saving figure", path)
        print("Called from mrd")
        plt.savefig(path, format='png', facecolor='k', edgecolor='k', dpi=300)
        plt.close()
    
def floored_percentage(val, digits):
        val *= 10 ** (digits + 2)
        return '{1:.{0}f}%'.format(digits, math.floor(val) / 10 ** digits)

def formatted_percentage(value, digits):
    format_str = "{:." + str(digits) + "%}"
    
    return format_str.format(value) 

def draw_original_vs_reconstructed_rim_z_score(x_true_img, x_hat_img, x_miss_img, plot_title, relative_error, observed_ratio, tcs, tcs_z_score, z_score, coord=None, folder=None, iteration=-1, time=-1):
        
    fig = plt.figure(frameon = False, figsize=(10,10))
    fig.set_size_inches(7, 7)
    
    #fig = plt.figure(frameon = False, figsize=(10,10))
    #fig.set_size_inches(7, 7)
    grid_rows = 3
    grid_cols = 1
    
    fg_color = 'white'
    bg_color = 'black'
    
    if plot_title:
        fig.suptitle(plot_title, color=fg_color, fontweight='normal', fontsize=10)
    
    if not coord:
        coord = find_xyz_cut_coords(x_true_img)
        
    grid = gridspec.GridSpec(grid_rows,grid_cols, hspace=0.2, wspace=0.2)
    
    subtitle = 'Original fMRI brain volume in three projections.'
    
    if time >-1: 
        subtitle = 'Original fMRI brain volume in three projections. Timepoint: ' + str(time + 1)
        
    main_ax = fig.add_subplot(grid[0, 0])
    main_ax.set_facecolor("blue")    
   
    main_ax.set_title(subtitle , color=fg_color, fontweight='normal', fontsize=8)  
    main_ax.set_aspect('equal')
      
   
    missing_ratio = (1.0 - observed_ratio)
    missing_ratio_str = formatted_percentage(missing_ratio, 2)
    print ("Missing Ratio Str:" + missing_ratio_str)                          
 
    relative_error_str = "{0:.5f}".format(relative_error) 
    relative_error_str = mf.format_number(relative_error, fmt='%1.2e')
    
    tsc_str = mf.format_number(tcs, fmt='%1.2e')
    tsc_z_score_str = mf.format_number(tcs_z_score, fmt='%1.2e')
    z_score_str = str(z_score)
                
    true_image = plotting.plot_epi(x_true_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = main_ax, cmap='jet', cut_coords=coord)     
    
    miss_ax = fig.add_subplot(grid[1, 0], sharex=main_ax)
    miss_ax.set_xlabel('(b)', color=bg_color)
    
    miss_ax.set_title('Corrupted fMRI brain volume. ' + " " + str("Missing Ratio: ")
                       + str(missing_ratio_str), color=fg_color, fontweight='normal', fontsize=8)
    
    miss_image = plotting.plot_epi(x_miss_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = miss_ax, cmap='jet', cut_coords=coord)  
    
    recov_ax = fig.add_subplot(grid[2, 0], sharex=main_ax)
    recov_ax.set_xlabel('(c)', color=bg_color)
    
    recov_ax.set_title('Completed. ' + " " + str("TCS: ") + tsc_str + " TCS(Z_Score >" + z_score_str + "): "  + tsc_z_score_str, color=fg_color, fontweight='normal', fontsize=8)
    
    recovered_image = plotting.plot_epi(x_hat_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = recov_ax, cmap='jet', cut_coords=coord)       
    
    if folder:
        fig_id =  str(folder) + "/" + "missing_ratio_" + str(missing_ratio_str)
    else:
        fig_id =  "_missing_ratio_" + str(missing_ratio_str)
        
    print ("Iteration: " + str(iteration))
    if iteration >=0:
        fig_id = fig_id[:-1] + '_' + str(iteration)
    else:
        fig_id = fig_id[:-1]
        
    if time >=0:
        fig_id = fig_id + '_timepoint_' + str(time)  

    save_fig_png(fig_id)