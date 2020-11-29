import texfig
import tensorly as tl

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
from datetime import datetime
import file_service as fs
import csv

import metric_util as mt
import data_util as du
import mri_draw_utils as mrd
import configparser
from os import path
import logging
import metadata as mdt
import tensor_util as tu
import tucker_tensor_completion as ct


config_loc = path.join('config')
config_filename = 'solution.config'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config

def complete_random_4D():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('random', 4)
    #root_dir = config.get('log', 'scratch.tucker.dir4D')
    root_dir = config.get('log', 'scratch.tucker.dir4Dmulti')
    #subject_meta_folder = fs.create_meta_directory(root_dir, "random")
    subject_meta_folder = "/work/scratch/tensor_completion/4D/tucker/multisubject/meta_subject2020-08-25_00_51_13/random"
    
    n = 1
    for i in range(n):
        observed_ratio_list = [0.9, 0.8, 0.7, 0.6, 0.65, 0.5, 0.4, 0.35, 0.3, 0.2, 0.1]
    #observed_ratio_list = [0.75,0.25]
        observed_ratio_list = [0.5]
        #solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
    observed_ratio_list = [0.5]
    subjects_root_dir = "/pl/mlsp/data/cobre/subjects/"
    subject_list = du.get_subjects(subjects_root_dir)
    
    subject_count = 0
    for subject_scan_path in subject_list:
        print(subject_scan_path)
        subject_count = subject_count + 1
        if subject_count > 67:
            break
        print ("Subject Count = " + str(subject_count))
        print("Processing Subject: " + str(subject_scan_path))
        
        subject_name= du.get_parent_name(subject_scan_path)   
        subject_meta_path = os.path.join(subject_meta_folder, subject_name)
        
        print ("Subject Name:" + subject_name  + "; Subject Metapath:" + subject_meta_path)
        
        accept_subject = False
        if os.path.exists(subject_meta_path):
            print ("Subject already exists!. Skip Processing." + subject_name + "; " + subject_scan_path)
            accept_subject = False
        else:
            print ("Accept subject." + subject_name)
            accept_subject = True
        
        du.delete_frames(subject_scan_path)
        if accept_subject:
            for item in observed_ratio_list:
                solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
                current_runner = ct.TuckerTensorCompletion(subject_scan_path, item, 4, 1, meta.logger, meta, subject_meta_folder)
                current_runner.complete()
    
        
if __name__ == "__main__":
    #tl.set_backend('tensorflow')
    #tfe.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)
    complete_random_4D()