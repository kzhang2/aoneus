import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
import sys, os
import pickle 
# import matplotlib.pyplot as plt
import time 
# from nerfacc import inclusive_prod, pack_info

def extract_fields(bound_min, bound_max, resolution, query_func, return_coords=False):
    N = 64
    X_coords = torch.linspace(bound_min[0], bound_max[0], resolution)
    Y_coords = torch.linspace(bound_min[1], bound_max[1], resolution)
    Z_coords = torch.linspace(bound_min[2], bound_max[2], resolution)
    X = X_coords.split(N)
    Y = Y_coords.split(N)
    Z = Z_coords.split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val

    if return_coords:
        return u, X_coords, Y_coords, Z_coords
    else:
        return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]

    return vertices, triangles

class NeuSRenderer:
    def __init__(self,
                 sdf_network,
                 deviation_network,
                 color_network,
                 base_exp_dir,
                 expID,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.base_exp_dir = base_exp_dir
        self.expID = expID

    def render_sonar_accel(self, dirs, pts, dists, ray_indices, arc_n_samples, cos_anneal_ratio=0.0):
        num_samples = len(ray_indices)
        num_ep = len(pts) - num_samples
        dirs_all = torch.cat([dirs[ray_indices], dirs], dim=0)
        pts_mid = pts + dirs_all * dists.reshape(-1, 1)/2
        sdf_network = self.sdf_network 
        deviation_network = self.deviation_network 
        color_network = self.color_network 

        sdf_nn_output = sdf_network(pts_mid)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]
        gradients = sdf_network.gradient(pts_mid).squeeze()

        # only evaluate at endpoints 
        sampled_color = color_network(pts_mid[num_samples:], gradients[num_samples:], dirs, feature_vector[num_samples:])
        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        inv_s = inv_s.expand(sdf.shape)

        true_cos = (dirs_all * gradients).sum(-1, keepdim=True)
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        transmittance = inclusive_prod(1-alpha[:num_samples, 0], indices=ray_indices)
        # print(len(ray_indices))
        packed_info = pack_info(ray_indices=ray_indices)
        packed_info = packed_info[packed_info[:, 1] > 0] 
        transmittance_inds = packed_info[:, 0] + packed_info[:, 1] - 1
        transmittance_ray_inds = ray_indices[transmittance_inds]
        transmittance_ep = torch.ones((num_ep,))
        transmittance_ep[transmittance_ray_inds] = transmittance[transmittance_inds]
        alpha_ep = alpha[num_samples:]
        weights = alpha_ep * transmittance_ep[:, None]
        intensity = weights * sampled_color  
        intensity = intensity.reshape(-1, arc_n_samples).sum(dim=1)
        weight_sum = weights.reshape(-1, arc_n_samples).sum(dim=1)

        gradient_error = (torch.linalg.norm(gradients, ord=2,
                                            dim=-1) - 1.0) ** 2

        return {
            "color_fine": intensity, 
            "weight_sum": weight_sum,
            "gradient_error": gradient_error,
        }



    def render_core_sonar(self,
                        dirs,
                        pts,
                        dists,
                        sdf_network,
                        deviation_network,
                        color_network,
                        n_pixels,
                        arc_n_samples,
                        ray_n_samples,
                        cos_anneal_ratio=0.0,
                        render_mode=False):
        
        pts_mid = pts + dirs * dists.reshape(-1, 1)/2

        if render_mode:
            with torch.no_grad():
                sdf_nn_output = sdf_network(pts_mid)
        else:
            sdf_nn_output = sdf_network(pts_mid)
        sdf = sdf_nn_output[:, :1]

        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts_mid).squeeze()


        # optimize memory consumption of below? 
        # print(pts_mid.shape)
        if render_mode:
            with torch.no_grad():
                sampled_color = color_network(pts_mid, gradients, dirs, feature_vector).reshape(n_pixels, arc_n_samples, ray_n_samples)

                inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        else:
            sampled_color = color_network(pts_mid, gradients, dirs, feature_vector).reshape(n_pixels, arc_n_samples, ray_n_samples)

            inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)

        inv_s = inv_s.expand(n_pixels*arc_n_samples*ray_n_samples, 1)
        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        # why calculate the next/prev sdfs like this? to enforce "consistency"? 
        # this happens to be the opposite of what is done during original neus upscaling (approx cos from sdf differences)
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(n_pixels, arc_n_samples, ray_n_samples).clip(0.0, 1.0)

        cumuProdAllPointsOnEachRay = torch.cat([torch.ones([n_pixels, arc_n_samples, 1]), 1. - alpha + 1e-7], -1)
    
        cumuProdAllPointsOnEachRay = torch.cumprod(cumuProdAllPointsOnEachRay, -1)

        TransmittancePointsOnArc = cumuProdAllPointsOnEachRay[:, :, ray_n_samples-2]
        
        alphaPointsOnArc = alpha[:, :, ray_n_samples-1]

        weights = alphaPointsOnArc * TransmittancePointsOnArc 

        intensityPointsOnArc = sampled_color[:, :, ray_n_samples-1]

        summedIntensities = (intensityPointsOnArc*weights).sum(dim=1) 

        # Eikonal loss
        gradients = gradients.reshape(n_pixels, arc_n_samples, ray_n_samples, 3)

        gradient_error = (torch.linalg.norm(gradients, ord=2,
                                            dim=-1) - 1.0) ** 2

        variation_error = torch.linalg.norm(alpha, ord=1, dim=-1).sum()

        return {
            'color': summedIntensities,
            'intensityPointsOnArc': intensityPointsOnArc,
            'sdf': sdf,
            'prev_sdf': estimated_prev_sdf,
            'next_sdf': estimated_next_sdf,
            'alpha': alpha,
            'dists': dists,
            'gradients': gradients,
            's_val': 1.0 / inv_s,
            'weights': weights,
            'cdf': c.reshape(n_pixels, arc_n_samples, ray_n_samples),
            'gradient_error': gradient_error,
            'variation_error': variation_error
        }

    def render_sonar(self, rays_d, pts, dists, n_pixels,
                     arc_n_samples, ray_n_samples, ray_indices, cos_anneal_ratio=0.0,
                     render_mode=False):
        # Render core
        
        if ray_indices is None: 
            ret_fine = self.render_core_sonar(rays_d,
                                            pts,
                                            dists,
                                            self.sdf_network,
                                            self.deviation_network,
                                            self.color_network,
                                            n_pixels,
                                            arc_n_samples,
                                            ray_n_samples,
                                            cos_anneal_ratio=cos_anneal_ratio,
                                            render_mode=render_mode)
            
            color_fine = ret_fine['color']
            weights = ret_fine['weights']
            weights_sum = weights.sum(dim=-1, keepdim=True)
            gradients = ret_fine['gradients']
            #s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

            return {
                'color_fine': color_fine,
                'weight_sum': weights_sum,
                'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
                'gradients': gradients,
                'weights': weights,
                'intensityPointsOnArc': ret_fine["intensityPointsOnArc"],
                'gradient_error': ret_fine['gradient_error'],
                'variation_error': ret_fine['variation_error'],
                "sdf": ret_fine["sdf"],
                "prev_sdf": ret_fine["prev_sdf"],
                "next_sdf": ret_fine["next_sdf"],
                "alpha": ret_fine["alpha"],
            }
        else:
            ret_fine = self.render_sonar_accel(rays_d, pts, dists, ray_indices, arc_n_samples)
            # return {
            #     "color_fine": ret_fine["color"],
            #     "weight_sum": ret_fine["weight_sum"], 
            # }
            return ret_fine

    

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
