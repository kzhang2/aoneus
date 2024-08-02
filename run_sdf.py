import os, sys
import numpy as np
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import scipy.io
from helpers import *
from MLP import *
#from PIL import Image
import cv2 as cv
import time
import random
import string 
from pyhocon import ConfigFactory
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
import trimesh
from itertools import groupby
from operator import itemgetter
from load_data import *
import logging
import argparse 
# import wandb
# from models.testing import RenderNetLamb
import shutil

import NeuS.exp_runner

from math import ceil

# set seeds
torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

# def scatterXYZ(X,Y,Z, name, expID, elev=None, azim=None, lims=None):
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')  
#     #surf = ax.plot_trisurf(X, Y, Z, linewidth=0, antialiased=False)
#     if lims is not None:
#         ax.set_xlim(lims['x_min'], lims['x_max'])
#         ax.set_ylim(lims['y_min'], lims['y_max'])
#         ax.set_zlim(lims['z_min'], lims['z_max'])
#     surf = ax.scatter3D(X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().numpy(), color = "green")
#     if elev is not None and azim is not None:
#         print("Setting elevation and azimuth to {} {}".format(elev, azim))
#         ax.view_init(elev=elev, azim=azim)
#     plt.xlabel('x', fontsize=18)
#     plt.ylabel('y', fontsize=16)
#     plt.savefig("./experiments/{}/figures/scatters/{}.png".format(expID, name))
#     plt.clf()

def make_occ_eval_fn(neusis_runner, render_step_size=0.05):
    def occ_eval_fn(x):
        with torch.no_grad():
            # print(x.shape)
            sdf = neusis_runner.sdf_network.sdf(x)
            inv_s = neusis_runner.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)

            estimated_next_sdf = sdf - render_step_size * 0.5
            estimated_prev_sdf = sdf + render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            return alpha
    return occ_eval_fn

class Runner:
    def __init__(self, conf, is_continue=False, write_config=True, testing=False, neus_conf=None, use_wandb=True, random_seed=0):
        conf_path = conf
        f = open(conf_path)
        conf_text = f.read()
        conf_name = str(conf_path).split("/")[-1][:-5]
        self.is_continue = is_continue
        self.conf = ConfigFactory.parse_string(conf_text)
        if use_wandb:
            project_name = "testing" if testing else "testing"
            # breakpoint()
            run = wandb.init(
                # Set the project where this run will be logged
                project=project_name,
                # Track hyperparameters and run metadata
                config=self.conf.as_plain_ordered_dict(),
                name=f"{conf_name}-{str(random_seed)}",
                dir="/tmp/"
            )
        self.neus_conf = neus_conf
        self.write_config = write_config
        if random_seed > 0:
            torch.random.manual_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)
        self.random_seed = random_seed
        self.use_wandb = use_wandb
        self.conf_path = conf_path

    def set_params(self):
        self.expID = self.conf.get_string('conf.expID') 

        dataset = self.conf.get_string('conf.dataset')
        self.image_setkeyname =  self.conf.get_string('conf.image_setkeyname') 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.N_rand = self.conf.get_int('train.num_select_pixels') #H*W 
        self.arc_n_samples = self.conf.get_int('train.arc_n_samples')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.percent_select_true = self.conf.get_float('train.percent_select_true', default=0.5)
        self.r_div = self.conf.get_bool('train.r_div')
        self.train_frac = self.conf.get_float("train.train_frac", default=1.0)
        self.accel = self.conf.get_bool('train.accel', default=False)
        # breakpoint()
        self.val_img_freq = self.conf.get_int("train.val_img_freq", default=10000)
        self.lamb_shading = self.conf.get_bool("train.lamb_shading", default=False)
        self.do_weight_norm = self.conf.get_bool("train.do_weight_norm", default=False)
        self.mode_tradeoff_schedule = self.conf.get_string("train.mode_tradeoff_schedule", default="none")
        self.mode_tradeoff_step_iter = self.conf.get_int("train.mode_tradeoff_step_iter", default=-1)
        self.rgb_weight = self.conf.get_float("train.rgb_weight", default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.variation_reg_weight = self.conf.get_float('train.variation_reg_weight')
        self.px_sample_min_weight = self.conf.get_float('train.px_sample_min_weight')
        # TODO: make below more reasonable? 
        self.weight_sum_factor = self.conf.get_float("train.weight_sum_factor", default=0.1)
        self.dark_weight_sum_factor = self.conf.get_float("train.dark_weight_sum_factor", default=0.0)

        self.ray_n_samples = self.conf['model.neus_renderer']['n_samples']
        # TODO: make below more flexible
        self.base_exp_dir = f"{self.expID}/{self.random_seed}"
        os.makedirs(self.base_exp_dir, exist_ok=True)
        shutil.copy(self.conf_path, f"{self.base_exp_dir}/config.conf")
        self.randomize_points = self.conf.get_float('train.randomize_points')
        self.select_px_method = self.conf.get_string('train.select_px_method')
        self.select_valid_px = self.conf.get_bool('train.select_valid_px')        
        self.x_max = self.conf.get_float('mesh.x_max')
        self.x_min = self.conf.get_float('mesh.x_min')
        self.y_max = self.conf.get_float('mesh.y_max')
        self.y_min = self.conf.get_float('mesh.y_min')
        self.z_max = self.conf.get_float('mesh.z_max')
        self.z_min = self.conf.get_float('mesh.z_min')
        self.level_set = self.conf.get_float('mesh.level_set')

        self.data = load_data(dataset)

        self.H, self.W = self.data[self.image_setkeyname][0].shape

        self.r_min = self.data["min_range"]
        self.r_max = self.data["max_range"]
        self.phi_min = -self.data["vfov"]/2
        self.phi_max = self.data["vfov"]/2
        self.vfov = self.data["vfov"]
        self.hfov = self.data["hfov"]


        self.cube_center = torch.Tensor([(self.x_max + self.x_min)/2, (self.y_max + self.y_min)/2, (self.z_max + self.z_min)/2])

        self.timef = self.conf.get_bool('conf.timef')
        self.end_iter = self.conf.get_int('train.end_iter')
        self.start_iter = self.conf.get_int('train.start_iter')
         
        self.object_bbox_min = self.conf.get_list('mesh.object_bbox_min')
        self.object_bbox_max = self.conf.get_list('mesh.object_bbox_max')

        r_increments = []
        self.sonar_resolution = (self.r_max-self.r_min)/self.H
        for i in range(self.H):
            r_increments.append(i*self.sonar_resolution + self.r_min)

        self.r_increments = torch.FloatTensor(r_increments).to(self.device)

        # extrapath = './experiments/{}'.format(self.expID)
        # if not os.path.exists(extrapath):
        #     os.makedirs(extrapath)

        # extrapath = './experiments/{}/checkpoints'.format(self.expID)
        # if not os.path.exists(extrapath):
        #     os.makedirs(extrapath)

        # extrapath = './experiments/{}/model'.format(self.expID)
        # if not os.path.exists(extrapath):
        #     os.makedirs(extrapath)

        # if self.write_config:
        #     with open('./experiments/{}/config.json'.format(self.expID), 'w') as f:
        #         json.dump(self.conf.__dict__, f, indent = 2)

        # Create all image tensors beforehand to speed up process

        self.i_train = np.arange(len(self.data[self.image_setkeyname]))

        self.coords_all_ls = [(x, y) for x in np.arange(self.H) for y in np.arange(self.W)]
        self.coords_all_set = set(self.coords_all_ls)

        #self.coords_all = torch.from_numpy(np.array(self.coords_all_ls)).to(self.device)

        self.del_coords = []
        for y in np.arange(self.W):
            tmp = [(x, y) for x in np.arange(0, self.ray_n_samples)]
            self.del_coords.extend(tmp)

        self.coords_all = list(self.coords_all_set - set(self.del_coords))
        self.coords_all = torch.LongTensor(self.coords_all).to(self.device)

        self.criterion = torch.nn.L1Loss(reduction='sum')
        
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)

        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        if self.neus_conf is not None:
            neus_runner = NeuS.exp_runner.Runner(self.neus_conf, init_opt=False, sdf_network=self.sdf_network, random_seed=self.random_seed)
            params_to_train += list(neus_runner.nerf_outside.parameters())
            params_to_train += list(neus_runner.deviation_network.parameters()) 
            params_to_train += list(neus_runner.color_network.parameters())  
            self.neus_runner = neus_runner

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)


        self.iter_step = 0
        self.renderer = NeuSRenderer(self.sdf_network,
                                    self.deviation_network,
                                    self.color_network if not self.lamb_shading else RenderNetLamb(),
                                    self.base_exp_dir,
                                    self.expID,
                                    **self.conf['model.neus_renderer'])  

        latest_model_name = None
        if self.is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth': #and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            ckpt_dir = os.path.join(self.base_exp_dir, 'checkpoints')
            self.load_checkpoint(f"{ckpt_dir}/{latest_model_name}")

        if self.accel:
            self.occ_eval_fn = make_occ_eval_fn(self) 
            grid_resolution = 128
            grid_nlvl = 1
            device = torch.device("cuda:0")
            aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
            self.estimator = OccGridEstimator(
                roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
            ).cuda()
        else:
            self.estimator = None


    def plotAllArcs(self, use_new=True):
        # This function is used to plot all arc points from all images in the same reference frame
        # this is to verify that we get an approximate shape and that the coordinate transformation 
        # is correct
        i_train = np.arange(len(self.data[self.image_setkeyname]))

        all_points = []
        for j in trange(0, len(i_train)):
            img_i = i_train[j]
            # print(img_i)
            target = self.data[self.image_setkeyname][img_i]
            target = torch.Tensor(target).to(self.device)
            pose = self.data["sensor_poses"][img_i]
            c2w = torch.tensor(pose).cuda().float()
            coords = torch.nonzero(target)
            n_pixels = len(coords)
            if n_pixels == 0: continue
            # TODO: do something faster than below 
            # essentially, just get dirs before normalization from the get_arcs function 

            # old
            if not use_new:
                _, _, _, _, pts, _ = get_arcs(self.H, self.W, self.phi_min, self.phi_max, self.r_min, self.r_max,  torch.Tensor(pose), n_pixels,
                                                        self.arc_n_samples, self.ray_n_samples, self.hfov, coords,
                                                        self.r_increments, self.randomize_points, 
                                                        self.device, self.cube_center)
                pts = pts.reshape(n_pixels, self.arc_n_samples, self.ray_n_samples, 3)
                pts = pts[:, :, -1, :]
                # print(pts.shape)
                all_points.append(pts)
            else:
                img_y = coords[:, 0] # img y coords
                img_x = coords[:, 1] # img x coords 
                phi = (
                    torch.linspace(self.phi_min, self.phi_max, self.arc_n_samples)
                    .float()
                    .repeat(n_pixels)
                    .reshape(n_pixels, -1)
                )
                sonar_resolution = (self.r_max - self.r_min) / self.H
                # compute radius at each pixel
                r = img_y * sonar_resolution + self.r_min
                # compute bearing angle at each pixel (azimuth)
                theta = -self.hfov / 2 + img_x * self.hfov / self.W
                coords = torch.stack(
                    (
                        r.repeat_interleave(self.arc_n_samples).reshape(n_pixels, -1),
                        theta.repeat_interleave(self.arc_n_samples).reshape(n_pixels, -1),
                        phi,
                    ),
                    dim=-1,
                )
                coords = coords.reshape(-1, 3)
                X = coords[:, 0] * torch.cos(coords[:, 1]) * torch.cos(coords[:, 2])
                Y = coords[:, 0] * torch.sin(coords[:, 1]) * torch.cos(coords[:, 2])
                Z = coords[:, 0] * torch.sin(coords[:, 2])
                pts = c2w @ torch.stack((X, Y, Z, torch.ones_like(X)))
                pts = pts[:3, ...].T

                all_points.append(pts)
        
        all_points = torch.cat(all_points, dim=0)
        return all_points, target
    
    def getRandomImgCoordsByPercentage(self, target):
        # 1. replace below with torch sampling + masking (double check it still works)
        # 2. dilate mask ot true locations, so dark areas around bright areas are also preferentially
        #    sampled 
        true_coords = []
        for y in np.arange(self.W):
            col = target[:, y]
            gt0 = col > 0
            indTrue = np.where(gt0)[0]
            if len(indTrue) > 0:
                true_coords.extend([(x, y) for x in indTrue])
        sampling_perc = int(self.percent_select_true*len(true_coords))
        true_coords = random.sample(true_coords, sampling_perc)
        true_coords = list(set(true_coords) - set(self.del_coords))
        true_coords = torch.LongTensor(true_coords).to(self.device)
        target = torch.Tensor(target).to(self.device)
        if self.iter_step%len(self.data[self.image_setkeyname]) !=0:
            N_rand = 0
        else:
            N_rand = self.N_rand
        N_rand = self.N_rand
        coords = select_coordinates(self.coords_all, target, N_rand, self.select_valid_px)
        coords = torch.cat((coords, true_coords), dim=0)
            
        return coords, target

    def train(self):
        loss_arr = []
        self.validate_mesh(threshold = self.level_set)

        # make train/validation sets
        # fix validation set for fair comparisons 
        # i_all = np.arange(len(self.data[self.image_setkeyname]))
        i_train = [] 
        i_val = []
        if self.train_frac < 1.0: 
            val_skip = int(1 / (1 - self.train_frac))
            for i in range(len(self.data[self.image_setkeyname])):
                if i % val_skip == 0:
                    i_val.append(i)
                else:
                    i_train.append(i)
        else:
            i_train = np.arange(len(self.data[self.image_setkeyname]))
            i_val = [] 
        # np.random.shuffle(i_all)
        # split_ind = int(self.train_frac * len(i_all))
        # i_train = i_all[:split_ind]
        # i_val = i_all[split_ind:]

        for i in trange(self.start_iter, self.end_iter, len(self.data[self.image_setkeyname])):
            # i_train = np.arange(len(self.data[self.image_setkeyname]))
            np.random.shuffle(i_train)
            loss_total = 0
            sum_intensity_loss = 0
            sum_eikonal_loss = 0
            sum_total_variational = 0
            sum_neus_loss = 0 
            
            for j in trange(0, len(i_train)):
                if self.accel:
                    self.estimator.update_every_n_steps(step=self.iter_step, occ_eval_fn=self.occ_eval_fn)
                log_dict = {}
                img_i = i_train[j]
                target = self.data[self.image_setkeyname][img_i]

                
                pose = self.data["sensor_poses"][img_i]  
                
                if self.select_px_method == "byprob":
                    coords, target = self.getRandomImgCoordsByProbability(target)
                else:
                    coords, target = self.getRandomImgCoordsByPercentage(target)

                n_pixels = len(coords)
                # print(n_pixels)

                # r holds radius per sample if estimator is none, otherwise it is  nONe
                rays_d, dphi, r, rs, pts, dists = get_arcs(self.H, self.W, self.phi_min, self.phi_max, self.r_min, self.r_max,  torch.Tensor(pose), n_pixels,
                                                        self.arc_n_samples, self.ray_n_samples, self.hfov, coords, self.r_increments, 
                                                        self.randomize_points, self.device, self.cube_center, self.estimator)

                
                target_s = target[coords[:, 0], coords[:, 1]]

                render_out = self.renderer.render_sonar(rays_d, pts, dists, n_pixels, 
                                                        self.arc_n_samples, self.ray_n_samples, r,
                                                        cos_anneal_ratio=self.get_cos_anneal_ratio())


                gradient_error = render_out['gradient_error'].reshape(-1, 1) #.reshape(n_pixels, self.arc_n_samples, -1)
                # gradient_error = torch.tensor(0)
                eikonal_loss = gradient_error.sum()*(1/gradient_error.shape[0])
                variation_regularization = render_out['variation_error']*(1/(self.arc_n_samples*self.ray_n_samples*n_pixels))
                # variation_regularization = torch.tensor(0)

                # try bright weight sum regularization 
                if self.weight_sum_factor > 0.0:
                    bright_weight_sums = render_out["weight_sum"][target_s > 0.0]
                    ones_target = torch.ones_like(bright_weight_sums) 
                    # modified with max
                    # weight_norm_loss = self.weight_sum_factor * torch.mean((torch.maximum(ones_target-bright_weight_sums, torch.zeros_like(ones_target)))**2)
                    weight_norm_loss = self.weight_sum_factor * torch.mean((ones_target-bright_weight_sums)**2)
                else:
                    weight_norm_loss = torch.tensor(0.0)

                # weight sparsity regularization 
                # bright_weights = render_out["weights"][target_s > 0.0]
                # weight_sparse_loss = 0.1 * torch.nn.functional.l1_loss(bright_weights, torch.zeros_like(bright_weights))
                weight_sparse_loss = 0.0

                # dark weight sum regularization
                # breakpoint()
                if self.dark_weight_sum_factor > 0.0:
                    dark_weights = render_out["weight_sum"][target_s == 0.0]
                    zeros_target = torch.zeros_like(dark_weights)
                    dark_weight_norm_loss = self.dark_weight_sum_factor * torch.mean((dark_weights - zeros_target)**2)
                else:
                    dark_weight_norm_loss = torch.tensor(0.0)

                if self.r_div:
                    intensityPointsOnArc = render_out["intensityPointsOnArc"]
                    intensity_fine = (torch.divide(intensityPointsOnArc, rs)*render_out["weights"]).sum(dim=1) 
                else:
                    intensity_fine = render_out['color_fine']

                if self.do_weight_norm:
                    if len(intensity_fine.shape) == 1:
                        intensity_fine = intensity_fine[:, None]
                    intensity_fine[target_s > 0.0] = intensity_fine[target_s > 0.0] / render_out["weight_sum"][target_s > 0.0]

                intensity_error = self.criterion(intensity_fine.squeeze(), target_s.squeeze())*(1/n_pixels)
                
                loss = intensity_error + eikonal_loss * self.igr_weight  + variation_regularization*self.variation_reg_weight
                loss += weight_norm_loss 
                loss += weight_sparse_loss
                loss += dark_weight_norm_loss
                if self.neus_conf is not None: 
                    if self.mode_tradeoff_schedule == "step":
                        if self.iter_step < self.mode_tradeoff_step_iter:
                            neus_loss = torch.tensor([0.]) 
                        else:
                            neus_loss = self.neus_runner.do_one_iter(img_i % self.neus_runner.dataset.n_images) 
                            loss = (1 - self.rgb_weight) * loss + self.rgb_weight * neus_loss
                    else:
                        neus_loss = self.neus_runner.do_one_iter(img_i % self.neus_runner.dataset.n_images) 
                        loss += neus_loss * 2 # TODO: fix this (add config?)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    lossNG = intensity_error + eikonal_loss * self.igr_weight 
                    loss_total += lossNG.cpu().numpy().item()
                    sum_intensity_loss += intensity_error.cpu().numpy().item()
                    sum_eikonal_loss += eikonal_loss.cpu().numpy().item()
                    sum_total_variational +=  variation_regularization.cpu().numpy().item()
                    if self.neus_conf is not None:
                        sum_neus_loss += neus_loss.cpu().numpy().item()
                
                
                self.iter_step += 1
                self.update_learning_rate()

                del(target)
                del(target_s)
                del(rays_d)
                del(pts)
                del(dists)
                del(render_out)
                del(coords)
                # break
                log_dict["sonar_intensity_loss"] = intensity_error.item()

                # end of epoch
                if j == len(i_train) - 1:
                    epoch_num = i // len(self.data[self.image_setkeyname]) # duplicated with below
                    log_dict["epoch_sonar_intensity_loss"] = sum_intensity_loss/len(i_train)
                    log_dict["epoch_num"] = epoch_num
                    if (epoch_num+1) % self.val_img_freq == 0:
                        tqdm.write("validation\n")
                        val_metric = 0
                        for i in trange(len(i_val)):
                            val_ind = i_val[i]
                            curr_img_val = render_image(self, val_ind, self.estimator)
                            curr_gt_val = self.data[self.image_setkeyname][val_ind]
                            val_metric += np.mean((curr_img_val - curr_gt_val) ** 2)
                        val_metric = val_metric / len(i_val) 
                        log_dict["mean_val_mse"] = val_metric

                        img = render_image(self, i_val[len(i_val)//2], self.estimator)
                        if self.use_wandb:
                            log_dict["val_vis"] = wandb.Image((np.clip(img, 0, 1)*255).astype(np.uint8))
                        img_train = render_image(self, i_train[len(i_train)//2], self.estimator)
                        if self.use_wandb:
                            log_dict["train_vis"] = wandb.Image((np.clip(img_train, 0, 1)*255).astype(np.uint8))
                        train_gt_img = self.data[self.image_setkeyname][i_train[len(i_train)//2]]
                        if self.use_wandb:
                            log_dict["train_gt_vis"] = wandb.Image((np.clip(train_gt_img, 0, 1)*255).astype(np.uint8))
                        gt_img = self.data[self.image_setkeyname][i_val[len(i_val)//2]]
                        if self.use_wandb:
                            log_dict["val_gt_vis"] = wandb.Image((np.clip(gt_img, 0, 1)*255).astype(np.uint8))
                        log_dict["epoch_num_val"] = epoch_num

                    # saving mesh + novel view synthesis for neus
                    if epoch_num == 0 or epoch_num % self.val_mesh_freq == 0:
                        mesh_path = self.validate_mesh(threshold = self.level_set)
                        if self.neus_conf is not None: 
                            # self.neus_runner.validate_mesh() 
                            self.neus_runner.validate_image()
                        if self.use_wandb:
                            log_dict["mesh_recon"] = wandb.Object3D(open(mesh_path))
                if self.use_wandb:
                    wandb.log(log_dict)

            with torch.no_grad():
                l = loss_total/len(i_train)
                iL =  sum_intensity_loss/len(i_train)
                eikL =  sum_eikonal_loss/len(i_train)
                varL =  sum_total_variational/len(i_train)
                if self.neus_conf is not None:
                    nl = sum_neus_loss / len(i_train)
                loss_arr.append(l)
            # breakpoint()
            epoch_num = i // len(self.data[self.image_setkeyname])

            # saving checkpoint
            if epoch_num == 0 or epoch_num % self.save_freq == 0:
                logging.info('iter:{} ********************* SAVING CHECKPOINT ****************'.format(self.optimizer.param_groups[0]['lr']))
                self.save_checkpoint()
                if self.neus_conf is not None: 
                    self.neus_runner.save_checkpoint()
            
            # write to terminal
            if epoch_num % self.report_freq == 0:
                report_str = f"iter:{self.iter_step:8>d} Loss={l} | intensity Loss={iL}  | eikonal loss={eikL} | total variation loss = {varL} | lr = {self.optimizer.param_groups[0]['lr']}"
                if self.neus_conf is not None: 
                    report_str = f"{report_str} | neus loss = {nl}"
                report_str = f"{report_str} | weight_norm_loss = {weight_norm_loss.item()}"
                report_str = f"{report_str} | dark_weight_norm_loss = {dark_weight_norm_loss.item()}"
                # report_str = f"{report_str} | weight_sparse_loss = {weight_sparse_loss.item()}"
                # print(report_str)
                tqdm.write(report_str)
        
        self.save_checkpoint()
        self.validate_mesh(threshold = self.level_set)


    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def load_checkpoint(self, checkpoint_name):
        # checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        checkpoint = torch.load(checkpoint_name, map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

    def update_learning_rate(self):
        if self.iter_step <= self.warm_up_end: # do i really need <=?
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        # breakpoint()
        bound_min = torch.tensor(self.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)

        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh_path = os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.obj'.format(self.iter_step))
        mesh.export(mesh_path)
        return mesh_path


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default="./confs/conf.conf")
    parser.add_argument('--neus_conf', type=str)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--disable_wandb", action="store_true")

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.is_continue, testing=args.testing, neus_conf=args.neus_conf, random_seed=args.random_seed, use_wandb=not args.disable_wandb)
    runner.set_params()
    runner.train()
