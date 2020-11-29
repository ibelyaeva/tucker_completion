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
import ellipsoid_mask as elm
import metric_util as mt


folder_path_subject1 = "/home/ec2-user/analysis/data/subject1"
data_path_subject1   = "swaAMAYER+cobre01_63001+M87100944+20110309at135133+RSTpre_V01_R01+CM.nii"

ellipsoid_masks_folder = "/work/pl/sch/analysis/results/masked_images/ellipsoid_masks"
ellipsoid_mask1_path = "size_20_7_15.nii"
ellipsoid_mask2_path = "size_20_11_25.nii"
ellipsoid_mask3_path = "size_35_20_25.nii"

def get_parent_name(file_path):
    current_dir_name = os.path.split(os.path.dirname(file_path))[1]
    return current_dir_name

def get_subjects(root_dir):
    subject_list = []
    for root,d_names,f_names in os.walk(root_dir):
        for f in f_names:
            if f.startswith("swa"):
                file_path = os.path.join(root, f)
                subject_list.append(file_path)
    return subject_list

def delete_volumes(file_path, scan_number):
       
    corrupted_volumes_list_scan_numbers = []
    replaced_frames = {}

    x_img = mt.read_image_abs_path(file_path)
    counter = 0    
    included_volumes_count = 0
    volumes_list = []

    for img in image.iter_img(x_img):
        print ("Volume Index: " + str(counter))
        if counter in scan_number:
            print ("Skipping volume: " + str(counter))
        else:
            print ("Adding volume to the list " + str(counter))
            volumes_list.append(img)
            included_volumes_count = included_volumes_count + 1
        counter = counter + 1
       
    x_img = image.concat_imgs(volumes_list)
    print ("Total Included Volumes = " + str(included_volumes_count) + "; Final Image = " + str(x_img.get_data().shape))
    return x_img

def delete_frames(file_path):

    scans = [0, 1, 2, 3, 4, 5]
    x_img = mt.read_image_abs_path(file_path)
    x_img_data = np.array(x_img.get_data())
    ts_count = x_img_data.shape[3]
    print(x_img)
    
    if (ts_count) > 144:
        print ("removing uncalibrated volumes " + file_path)
        x_updated_img = delete_volumes(file_path, scans)
        nib.save(x_updated_img , file_path)
    else:
        print ("skipping uncalibrated volumes")


def get_folder_subject1():
    folder_path_subject1 = "/pl/mlsp/data/subjects/subject1"
    return folder_path_subject1


def corrupted4D_10_frames_path():
    path = "/work/pl/sch/analysis/results/masked_images/d4/10/subject1_10.nii"
    return path

def corrupted4D_20_frames_path():
    path = "/work/pl/sch/analysis/results/masked_images/d4/20/subject1_20.nii"
    return path

def corrupted4D_50_frames_path():
    path = "/work/pl/sch/analysis/results/masked_images/d4/50/subject1_50.nii"
    return path

def get_path_subject1():
    data_path_subject1   = "cobre.nii"
    return data_path_subject1

def get_full_path_subject1():
    subject_abs_path  = os.path.join(get_folder_subject1(), get_path_subject1())
    return subject_abs_path

def get_ellipse_mask_img_20_7_15():
    ellipsoid = elm.EllipsoidMask(0, -18, 17, 20, 17, 15)
    path = os.path.join(ellipsoid_masks_folder, ellipsoid_mask1_path)
    mask_img = mt.read_image_abs_path(path)
    print("Ellipsoid :" + "; x_r =" + str(ellipsoid.x_r) + "; y_r=" + str(ellipsoid.y_r) + "; z_r = " + str(ellipsoid.z_r) + "; Volume =" + str(ellipsoid.volume))
    return mask_img

