import texfig

import torch
from torch.autograd import Variable
import tensorly as tl
from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.random import check_random_state

tl.set_backend('pytorch')

import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt
import metric_util as mt
import data_util as du
from nilearn import image
from tensorflow.python.util import nest
import copy
from nilearn import plotting
import mri_draw_utils as mrd
from scipy import optimize 
from nilearn.masking import compute_background_mask
from nilearn.masking import compute_epi_mask
from collections import OrderedDict
import pandas as pd
from scipy import stats
from nilearn.image import math_img
import cost_computation as cst
import tensor_util as tu
import nibabel as nib
import os
import metadata as mdt
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor
from tensorflow.python.eager.context import context, EAGER_MODE, GRAPH_MODE
from tensorly.random import check_random_state
import tensorly.backend as T


class TuckerTensorCompletionRunner(object):
    
    def __init__(self, ground_truth_img, ground_truth, tensor_shape, x_init, mask_indices, z_scored_mask, 
                 sparse_observation_org,
                 norm_sparse_observation,
                 x_init_tcs,
                 ten_ones, max_tt_rank, observed_ratio, epsilon, train_epsilon, backtrack_const, logger, meta, d, z_score = 2):
        
        self.ground_truth_img = ground_truth_img
        self.ground_truth = ground_truth
        self.tensor_shape = tensor_shape
        self.x_init = x_init
        self.mask_indices = mask_indices
        self.z_scored_mask = z_scored_mask
        self.sparse_observation_org = sparse_observation_org
        self.norm_sparse_observation = norm_sparse_observation
        self.x_init_tcs = x_init_tcs
        self.ten_ones = ten_ones
        self.max_tt_rank = max_tt_rank
        self.observed_ratio = observed_ratio
        self.missing_ratio = 1.0 - observed_ratio
        self.logger = logger
        self.meta = meta
        self.d = d
        self.z_score = z_score
                 
        self.epsilon = epsilon
        self.train_epsilon = train_epsilon
        
        self.penalty = 0.0001
        self.num_epochs = 200
        
        self.backtrack_const = backtrack_const
        self.init()
       
    def init(self):
        self.rse_cost_history = []
        self.train_cost_history = []
        self.train_adj_cost_history = []
        self.tcs_cost_history = []
        self.tcs_z_scored_history = []
        self.cost_history = []
        self.grad_history = []
        
        self.scan_mr_folder = self.meta.create_scan_mr_folder(self.missing_ratio)
        self.scan_mr_iteration_folder = self.meta.create_scan_mr_folder_iteration(self.missing_ratio)
        self.images_mr_folder_iteration = self.meta.create_images_mr_folder_iteration(self.missing_ratio)
        self.suffix = self.meta.get_suffix(self.missing_ratio)
        self.images_folder = self.meta.images_folder
        
        self.logger.info(self.scan_mr_iteration_folder)
        self.logger.info(self.suffix)
        self.init_variables()

    def complete(self):
    
        self.logger.info("Starting Tensor Completion. Tensor Dimension:" + str(len(self.ground_truth)) + "; Tensor Shape: " +              str(self.tensor_shape) + "; Max Rank: " + str(self.max_tt_rank) + "; Solution Tolerance (Gradient): " + str(self.epsilon) + "; Convergence Tolerance (Train: )" + str(self.train_epsilon) + "; Factors Penalty: " + str(self.penalty) + "; Max Iterations: " + str(self.num_epochs))
        
        self.init_algorithm()
        
        self.x_hat = mt.reconstruct2(np.array(self.X.numpy()), self.ground_truth, self.mask_indices)
        self.x_hat_img = mt.reconstruct_image_affine_d(self.ground_truth_img, self.x_hat, self.d, self.tensor_shape)
        
        # save initial solution and cost
        #self.save_solution_scans_iteration(self.suffix, self.scan_mr_iteration_folder, 0)
        self.save_cost_history()
        
        lr = 0.001 
        
        optimizer = torch.optim.Adam([self.core]+self.tensor_factors, lr=lr)
        
        i = 0
        while self.train_cost > self.epsilon:
    
            i = i + 1
            optimizer.zero_grad()
            
            rec = tl.tucker_to_tensor(self.core, self.tensor_factors)
            loss_value = 0.5*tl.norm(rec - tensor, 2)
            
            grad = rec*self.sparsity_mask_tf - self.sparse_observation_tf
            grad_norm = tl.norm(grad, 2)
            log.info("grad norm = " + str(grad_norm.data))
            
            for f in self.tensor_factors: 
                loss_value = loss_value + self.penalty*f.pow(2).sum()
                
            self.cost_history.append(loss_value.numpy())
            
            loss_value.backward()
            optimizer.step()
                  
            self.X = rec * (1 - self.sparsity_mask_tf) + self.X * self.sparsity_mask_tf
                      
            self.train_cost = min(1.0,(2.0*np.sqrt(loss_value.numpy()) / self.norm_sparse_observation.numpy()))
            
            rse_cost = ((tl.norm(rec - self.ground_truth,2)/tl.norm(self.ground_truth),2)).numpy()
            tsc_score = cst.tsc(np.array(rec.numpy()),self.ground_truth, self.ten_ones, self.mask_indices)
            tcs_z_score = tu.tsc_z_score(np.array(rec.numpy()),self.ground_truth,self.ten_ones, self.mask_indices, self.z_scored_mask)
    
            self.tcs_cost_history.append(tsc_score)
            self.train_cost_history.append(self.train_cost)
            self.grad_history.append(grad_norm)
            
            
            self.tcs_z_scored_history.append(tcs_z_score)
            self.rse_cost_history.append(rse_cost)
            
            self.tsc_score = self.tcs_cost_history[i]
            self.tcs_z_score = self.tcs_z_scored_history[i]
            
            #self.save_solution_scans(self.suffix, self.scan_mr_folder)
                               
            self.x_hat = mt.reconstruct2(np.array(rec.numpy()), self.ground_truth, self.mask_indices)
            self.x_hat_img = mt.reconstruct_image_affine_d(self.ground_truth_img, self.x_hat, self.d, self.tensor_shape)
            
            #if i % 10 == 0:
            #    self.save_solution_scans_iteration(self.suffix, self.scan_mr_iteration_folder, i)
                
            self.logger.info("Len TSC Score History: " + str(len(self.tcs_cost_history)))
            self.save_cost_history()
            
            self.logger.info("Current Iteration #: " + str(i))
            
            #if i >= self.num_epochs:
            #    self.logger.info("Maximum # of Iterations exceded. Max Iter: " + str(self.num_epochs))
            #    break
    
            if i > 1:
                diff_train = np.abs(self.train_cost_history[i] - self.train_cost_history[i - 1]) / np.abs(self.train_cost_history[i])
                self.logger.info("Epoch: " + str(i) + "; GradNorm: " + str(grad_norm) +  "; Diff Train: " + str(diff_train)  + "; Loss: " + str(loss_value.numpy())  + "; Train Cost: " + str(self.train_cost) + "; TCS Score: " + str(tsc_score)  + "; TCS Z Score: " 
                                  + str(tcs_z_score) + "; RSE Cost: " + str(rse_cost))
        
                if diff_train <= self.train_epsilon:
                    self.logger.info("Optimization Completed. Breaking after " + str(i) + " iterations" + "; Reason Relative Tolerance of Training Iterations Exceeded Trheshold: " + str(self.train_epsilon))
                    break
    
            else:
                self.logger.info("Epoch: " + str(i) + "; GradNorm: " + str(grad_norm) +  "; Loss: " + str(loss_value)  + "; Train Cost: " + str(self.train_cost) + "; TCS Score: " + str(tsc_score)  + "; TCS Z Score: " 
                                  + str(tcs_z_score) + "; RSE Cost: " + str(rse_cost))

            
        
        self.logger.info("Optimization Completed After Iterations #: " + str(i))
        print("Optimization Completed After Iterations #: " + str(i))
                     
        self.logger.info("Observed Ratio: " + str(self.observed_ratio))
        self.logger.info("Final TCS Z-Score: " + str(self.tcs_z_score))
        self.logger.info("Final TCS Score: " + str(self.tsc_score))
        
        self.save_solution_scans(self.suffix, self.scan_mr_folder)
        self.save_cost_history()
        
        self.logger.info("Done ...")
        print("Done ...")
    
    def switch_to(self, mode):
        ctx = context()._eager_context
        ctx.mode = mode
        ctx.is_eager = mode == EAGER_MODE
    
    def init_variables(self):
       
        self.title = str(self.d) + "D fMRI Tensor Completion"
        
        self.original_shape = self.ground_truth.shape
        self.target_shape = mt.get_target_shape(self.ground_truth, self.d)
        
        self.logger.info("D = " + str(self.d) + "; Original Shape: " + str(self.original_shape) + "; Target Shape: " + str(self.target_shape))
        
        #save sparse_observation_org before reshaping
        self.sparse_observation_org = copy.deepcopy(self.ground_truth)
        self.sparse_observation_org[self.mask_indices == 0] = 0.0
        
        self.sparse_observation = copy.deepcopy(self.ground_truth)
        self.sparse_observation [self.mask_indices == 0] = 0.0
            
        self.norm_sparse_observation = np.linalg.norm(self.sparse_observation)
        
        # save x_miss_img
        self.x_miss_img = mt.reconstruct_image_affine(self.ground_truth_img, self.sparse_observation_org)
        
        # dont' change ground truth image
        
        # reshape mask_indices
        if self.d == 3 or self.d == 2:
            self.mask_indices = mt.reshape_as_nD(self.mask_indices, self.d,self.target_shape)
            
        self.logger.info("Mask Indices Shape: " + str(self.mask_indices.shape))
        
        #create x_miss
        self.x_miss = np.array(self.sparse_observation_org)
        
        if self.d == 3 or self.d == 2:
            self.x_miss =  mt.reshape_as_nD(self.x_miss, self.d,self.target_shape)
            
        self.logger.info("Miss X Shape: " + str(self.x_miss.shape))
        
        #update ground_truth if needed
        if self.d == 3 or self.d == 2:
            self.ground_truth = mt.reshape_as_nD(self.ground_truth, self.d,self.target_shape)
            
        self.logger.info("Ground Truth Shape: " + str(self.ground_truth.shape))
        
        #reshape ten_ones
        if self.d == 3 or self.d == 2:
            self.ten_ones = mt.reshape_as_nD(self.ten_ones, self.d,self.target_shape)
            
        self.logger.info("Ten Ones Truth Shape: " + str(self.ten_ones.shape))
        
        # reshape self.x_init_tcs
        if self.d == 3 or self.d == 2:
            self.x_init_tcs = mt.reshape_as_nD(self.x_init_tcs, self.d,self.target_shape)
            
        self.logger.info("Init TCS Shape: " + str(self.x_init_tcs.shape))
        
        # reshape self.x_init
        if self.d == 3 or self.d == 2:
            self.x_init = mt.reshape_as_nD(self.x_init, self.d,self.target_shape)
            
        self.logger.info("X Init Shape: " + str(self.x_init.shape))
        
        
        # reshape self.z_scored_mask
        if self.d == 3 or self.d == 2:
            self.z_scored_mask = mt.reshape_as_nD(self.z_scored_mask, self.d,self.target_shape)
            
        self.logger.info("Z Score Mask Shape: " + str(self.z_scored_mask.shape))
        
        self.ground_truth_tf = tl.tensor(self.ground_truth, requires_grad=False)
        self.sparsity_mask_tf = tl.tensor(self.mask_indices, requires_grad=False)
        self.sparse_observation_tf = tl.tensor(self.sparse_observation, requires_grad=False)
        self.ten_ones_tf = tl.tensor(self.ten_ones, requires_grad=False)
        
        self.logger.info("TF Sparsity Mask Shape: " + str(self.sparsity_mask_tf.shape))
        self.logger.info("TF Sparse Observation Shape: " + str(self.sparse_observation_tf.shape)) 
        
        self.x_reconstr_init = mt.reconstruct2(self.x_init_tcs, self.ground_truth, self.mask_indices)
        self.tsc_score_init = cst.tsc(self.x_reconstr_init, self.ground_truth, self.ten_ones, self.mask_indices).astype('float32')
        self.logger.info("TCS Score Initial Value: " + str(self.tsc_score_init))
        
        self.init_cost()
        
    def init_cost(self):
        train_cost_init = cst.relative_error_omega(self.x_init_tcs, self.ground_truth, self.sparse_observation)
        tcs_z_score_init = tu.tsc_z_score(self.x_init_tcs,self.ground_truth,self.ten_ones, self.mask_indices, self.z_scored_mask)
        rse_cost_init = cst.relative_error(self.x_init_tcs,self.ground_truth)
        cost_init = cst.compute_loss_np(self.x_init, self.mask_indices, self.sparse_observation)
        
        train_adjust_cost_init = train_cost_init/2.0
        grad_norm_init = np.linalg.norm((self.x_init_tcs - self.ground_truth))
        
        self.train_adj_cost_history.append(train_adjust_cost_init)
        self.grad_history.append(grad_norm_init)
        
        self.train_cost = train_cost_init
        self.rse_cost_history.append(rse_cost_init)
        self.cost_history.append(cost_init)
        self.train_cost_history.append(train_cost_init)
        self.tcs_cost_history.append(self.tsc_score_init)
        self.tcs_z_scored_history.append(tcs_z_score_init)
        
    def create_factor(self, shape):
        factor = (2*np.random.random_sample(shape) - 1).astype('float32')
        norm_ground_factor = np.linalg.norm(factor)
        factor = tl.tensor(factor * (1./norm_ground_factor), requires_grad=True)
        return factor
        
    def init_algorithm(self):
        self.X = tl.tensor(self.x_init, requires_grad=True)
        self.X1 = self.x_init
        
        ranks = tu.get_tensor_shape_as_list(self.X)
        
        self.tensor_factors = []
        for i in range(tl.ndim(self.X)):
            item = self.create_factor((self.X.shape[i] , ranks[i]))
            self.logger.info("Factor Shape" + str(item.shape))
            self.tensor_factors.append(item)
            
        self.core = tl.tensor(self.x_init, requires_grad=True)

        
    def save_solution_scans_iteration(self, suffix, folder, iteration): 
        
        self.logger.info("Missing Ratio: " + str(self.missing_ratio))
        self.logger.info("Suffix: " + str(suffix))
        self.logger.info("Folder: " + str(folder))
        self.logger.info("Iteration: " + str(iteration))
        
        x_true_path = os.path.join(folder,"x_true_img_" + str(suffix) + '_' + str(iteration))
        x_hat_path = os.path.join(folder,"x_hat_img_" + str(suffix) + '_' + str(iteration))
        x_miss_path = os.path.join(folder,"x_miss_img_" + str(suffix) + '_' + str(iteration))
        
        self.logger.info("x_hat_path: " + str(x_hat_path))
        nib.save(self.x_hat_img, x_hat_path)
            
        self.logger.info("x_miss_path: " + str(x_miss_path))
        nib.save(self.x_miss_img, x_miss_path)
            
        self.logger.info("x_true_path: " + str(x_true_path))
        nib.save(self.ground_truth_img, x_true_path)
         
        mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(self.ground_truth_img, 0), image.index_img(self.x_hat_img,0), image.index_img(self.x_miss_img, 0), self.title + " Iteration: " + str(iteration),
                        self.tcs_cost_history[iteration], self.observed_ratio, self.tcs_cost_history[iteration], self.tcs_z_scored_history[iteration], 2, coord=None, folder=self.images_mr_folder_iteration, iteration=iteration)
        
        
    def save_solution_scans(self, suffix, folder): 
        
        self.logger.info("Missing Ratio: " + str(self.missing_ratio))
        x_true_path = os.path.join(folder,"x_true_img_" + str(suffix))
        x_hat_path = os.path.join(folder,"x_hat_img_" + str(suffix))
        x_miss_path = os.path.join(folder,"x_miss_img_" + str(suffix))
        
        self.logger.info("x_hat_path: " + str(x_hat_path))
        nib.save(self.x_hat_img, x_hat_path)
            
        self.logger.info("x_miss_path: " + str(x_miss_path))
        nib.save(self.x_miss_img, x_miss_path)
            
        self.logger.info("x_true_path: " + str(x_true_path))
        nib.save(self.ground_truth_img, x_true_path)
        
        mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(self.ground_truth_img, 0), image.index_img(self.x_hat_img,0), image.index_img(self.x_miss_img, 0), self.title,
                    self.tsc_score, self.observed_ratio, self.tsc_score, self.tcs_z_score, 2, coord=None, folder=self.images_folder, iteration = -1, time=0)
        
        mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(self.ground_truth_img, 69), image.index_img(self.x_hat_img,69), image.index_img(self.x_miss_img, 69), self.title,
                    self.tsc_score, self.observed_ratio, self.tsc_score, self.tcs_z_score, 2, coord=None, folder=self.images_folder, iteration = -1, time=69)
        
        mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(self.ground_truth_img, 119), image.index_img(self.x_hat_img,119), image.index_img(self.x_miss_img, 119), self.title,
                    self.tsc_score, self.observed_ratio, self.tsc_score, self.tcs_z_score, 2, coord=None, folder=self.images_folder, iteration = -1, time = 119)
        
        mrd.draw_original_vs_reconstructed_rim_z_score(image.index_img(self.ground_truth_img, 143), image.index_img(self.x_hat_img,143), image.index_img(self.x_miss_img, 143), self.title,
                    self.tsc_score, self.observed_ratio, self.tsc_score, self.tcs_z_score, 2, coord=None, folder=self.images_folder, iteration = -1, time=143)
        
        
    def save_cost_history(self):
           
        output_cost = OrderedDict()
        indices = []

        cost_arr = []
        tsc_arr = []
        tsc_z_score_arr = []
        
        rse_arr = []
        
        grad_arr = []

        counter = 0
        for item in  self.cost_history:
            self.logger.info(item)
            cost_arr.append(item)
            indices.append(counter)
            counter = counter + 1
    
        output_cost['k'] = indices
        output_cost['cost'] = cost_arr
    
        output_df = pd.DataFrame(output_cost, index=indices)

        results_folder = self.meta.results_folder
        
        fig_id = 'solution_cost' + '_' + self.suffix
        mrd.save_csv_by_path(output_df, results_folder, fig_id)  

        tsc_score_output = OrderedDict()
        tsc_score_indices = []

        counter = 0
        for item in self.tcs_cost_history:
            tsc_arr.append(item)
            tsc_score_indices.append(counter)
            counter = counter + 1

        tsc_score_output['k'] = tsc_score_indices
        tsc_score_output['tsc_cost'] = tsc_arr
    
        output_tsc_df = pd.DataFrame(tsc_score_output, index=tsc_score_indices)
        fig_id = 'tsc_cost' + '_' + self.suffix
        mrd.save_csv_by_path(output_tsc_df, results_folder, fig_id) 
        
        # output z-score
        tsc_z_score_output = OrderedDict()
        tsc_z_score_indices = []
        
        counter = 0
        for item in self.tcs_z_scored_history:
            tsc_z_score_arr.append(item)
            tsc_z_score_indices.append(counter)
            counter = counter + 1

        tsc_z_score_output['k'] = tsc_z_score_indices
        tsc_z_score_output['tsc_z_cost'] = tsc_z_score_arr
        
        output_tsc_z_df = pd.DataFrame( tsc_z_score_output, index=tsc_z_score_indices)
        fig_id = 'tsc_z_cost' + '_' + self.suffix
        mrd.save_csv_by_path(output_tsc_z_df, results_folder, fig_id) 
        
        # output rse history
        
        rse_output = OrderedDict()
        rse_indices = []
        counter = 0
        
        for item in self.rse_cost_history:
            rse_arr.append(item)
            rse_indices.append(counter)
            counter = counter + 1

        rse_output['k'] = rse_indices
        rse_output['rse_cost'] = rse_arr
        
        output_rse_df = pd.DataFrame( rse_output, index=rse_indices)
        fig_id = 'rse_cost' + '_' + self.suffix
        mrd.save_csv_by_path(output_rse_df, results_folder, fig_id) 
        
        # output train history
        train_arr = []
        train_output = OrderedDict()
        train_indices = []
        counter = 0
        
        for item in self.train_cost_history:
            train_arr.append(item)
            train_indices.append(counter)
            counter = counter + 1

        train_output['k'] = train_indices
        train_output['train_cost'] = train_arr
        
        output_train_df = pd.DataFrame(train_output, index=train_indices)
        
        # save grad_norm
        grad_norm_arr = []
        grad_norm_output = OrderedDict()
        grad_norm_indices = []
        counter = 0
        
        for item in self.grad_history:
            grad_norm_arr.append(item)
            grad_norm_indices.append(counter)
            counter = counter + 1

        grad_norm_output['k'] = grad_norm_indices
        grad_norm_output['grad_norm'] = grad_norm_arr
        
        grad_norm_df = pd.DataFrame(grad_norm_output, index=train_indices)
        
        fig_id = 'train_cost_adj' + '_' + self.suffix
        mrd.save_csv_by_path(grad_norm_df, results_folder, fig_id)
        
        
        
        
        