import torch
# import matplotlib
import numpy as np

# matplotlib.use("Agg")
from MLP import *


torch.autograd.set_detect_anomaly(True)


def update_lr(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        if param_group["lr"] > 0.0000001:
            param_group["lr"] = param_group["lr"] * lr_decay
            learning_rate = param_group["lr"]
            print("learning r ate is updated to ", learning_rate)
    return 0


def save_model(expID, model, i):
    # save model
    model_name = "./experiments/{}/model/epoch.pt".format(expID)
    torch.save(model, model_name)
    return 0


def render_image(neusis_runner, pose_ind, estimator=None, debug=False):
    H = neusis_runner.H 
    W = neusis_runner.W 

    phi_min = neusis_runner.phi_min
    phi_max = neusis_runner.phi_max

    tx = torch.linspace(0, W - 1, W)
    ty = torch.linspace(0, H - 1, H)
    # need to use xy indexing to be consistent with render_image_from_rays
    pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing="xy")
    px = torch.stack([pixels_y, pixels_x], dim=-1)  # W, H, 2
    px = px.reshape(-1, 2).long() # int conversion needed 


    c2w = torch.from_numpy(neusis_runner.data["sensor_poses"][pose_ind]).cuda()
    r_min = neusis_runner.r_min
    r_max = neusis_runner.r_max
    n_selected_px = H * W
    arc_n_samples = neusis_runner.arc_n_samples
    ray_n_samples = neusis_runner.ray_n_samples
    hfov = neusis_runner.hfov
    r_increments = []
    sonar_resolution = (r_max - r_min) / H
    # print(sonar_resolution)
    for i in range(H):
        r_increments.append(i * sonar_resolution + r_min)
    r_increments = torch.tensor(r_increments).cuda()
    randomize_points = False
    device = "cuda:0"
    cube_center = neusis_runner.cube_center.cuda()

    dirs, dphi, r, rs, pts_r_rand, dists = get_arcs(
        H,
        W,
        phi_min,
        phi_max,
        r_min,
        r_max,
        c2w,
        n_selected_px,
        arc_n_samples,
        ray_n_samples,
        hfov,
        px,
        r_increments,
        randomize_points,
        device,
        cube_center,
        estimator=estimator
    )
    if estimator is not None: 
        ray_indices = r
    final_out = np.zeros((H, W))
    # weight_sum_out = np.zeros((H, W))
    # sdf_vals = []

    if estimator is None:
    # render a row at a time
        for i in range(H):
            curr_dirs = dirs[W*i*arc_n_samples*ray_n_samples:W*(i+1)*arc_n_samples*ray_n_samples]
            curr_pts_r_rand = pts_r_rand[W*i*arc_n_samples*ray_n_samples:W*(i+1)*arc_n_samples*ray_n_samples]
            curr_dists = dists[W*i*arc_n_samples:W*(i+1)*arc_n_samples]
            out = neusis_runner.renderer.render_sonar(curr_dirs, curr_pts_r_rand, curr_dists, W, arc_n_samples, ray_n_samples, r, neusis_runner.get_cos_anneal_ratio())
            curr_pixels = out["color_fine"].reshape(W).detach().cpu().numpy()
            # weight_sum_out[i] = out["weight_sum"].reshape(W).detach().cpu().numpy()
            # sdf_vals.append(out["alpha"].detach())
            del out

            final_out[i] = curr_pixels
    else:
        out = neusis_runner.renderer.render_sonar_accel(dirs, pts_r_rand, dists, ray_indices, arc_n_samples, neusis_runner.get_cos_anneal_ratio())
        final_out = out["color_fine"].reshape(H, W).detach().cpu().numpy()

    if debug:
        if estimator is not None:
            return final_out, pts_r_rand, ray_indices 
        else:
            return final_out, pts_r_rand
    else:
        return final_out

def get_arcs(
    H,
    W,
    phi_min,
    phi_max,
    r_min,
    r_max,
    c2w,
    n_selected_px,
    arc_n_samples,
    ray_n_samples,
    hfov,
    px,
    r_increments,
    randomize_points,
    device,
    cube_center,
    estimator=None,
):
    i = px[:, 0] # img y coords
    j = px[:, 1] # img x coords 

    # sample angle phi (elevation)
    phi = (
        torch.linspace(phi_min, phi_max, arc_n_samples)
        .float()
        .repeat(n_selected_px)
        .reshape(n_selected_px, -1)
    )

    dphi = (phi_max - phi_min) / arc_n_samples
    rnd = -dphi + torch.rand(n_selected_px, arc_n_samples) * 2 * dphi

    sonar_resolution = (r_max - r_min) / H
    if randomize_points:
        phi = torch.clip(phi + rnd, min=phi_min, max=phi_max)

    # compute radius at each pixel
    r = i * sonar_resolution + r_min
    # compute bearing angle at each pixel (azimuth)
    theta = -hfov / 2 + j * hfov / W

    # Need to calculate coords to figure out the ray direction
    # the following operations mimick the cartesian product between the two lists [r, theta] and phi
    # coords is of size: n_selected_px x arc_n_samples x 3
    coords = torch.stack(
        (
            r.repeat_interleave(arc_n_samples).reshape(n_selected_px, -1),
            theta.repeat_interleave(arc_n_samples).reshape(n_selected_px, -1),
            phi,
        ),
        dim=-1,
    )
    coords = coords.reshape(-1, 3)
    # Transform to cartesian to apply pose transformation and get the direction
    # transformation as described in https://www.ri.cmu.edu/pub_files/2016/5/thuang_mastersthesis.pdf
    X = coords[:, 0] * torch.cos(coords[:, 1]) * torch.cos(coords[:, 2])
    Y = coords[:, 0] * torch.sin(coords[:, 1]) * torch.cos(coords[:, 2])
    Z = coords[:, 0] * torch.sin(coords[:, 2])

    dirs = torch.stack((X, Y, Z, torch.ones_like(X))).T
    dirs = torch.matmul(c2w, dirs.T).T
    origin = torch.matmul(c2w, torch.tensor([0.0, 0.0, 0.0, 1.0])).unsqueeze(dim=0)
    dirs = dirs - origin
    dirs = dirs[:, 0:3]
    dirs = torch.nn.functional.normalize(dirs, dim=1)

    if estimator is None:
        dirs = dirs.repeat_interleave(ray_n_samples, 0)

        holder = torch.empty(
            n_selected_px, arc_n_samples * ray_n_samples, dtype=torch.long
        ).to(device)
        bitmask = torch.zeros(ray_n_samples, dtype=torch.bool) # where end points of rays are
        bitmask[ray_n_samples - 1] = True
        bitmask = bitmask.repeat(arc_n_samples)

        # I think this for loop is slow in particular 
        for n_px in range(n_selected_px):
            holder[n_px, :] = torch.randint(
                0, i[n_px] + 1, (arc_n_samples * ray_n_samples,) # already excludes right endpoint (bug?)
            )
            holder[n_px, bitmask] = i[n_px]

        holder = holder.reshape(n_selected_px, arc_n_samples, ray_n_samples)

        holder, _ = torch.sort(holder, dim=-1)

        holder = holder.reshape(-1)

        r_samples = torch.index_select(r_increments, 0, holder).reshape(
            n_selected_px, arc_n_samples, ray_n_samples
        )

        rnd = torch.rand((n_selected_px, arc_n_samples, ray_n_samples)) * sonar_resolution

        if randomize_points:
            r_samples = r_samples + rnd

        rs = r_samples[:, :, -1]
        r_samples = r_samples.reshape(n_selected_px * arc_n_samples, ray_n_samples)

        theta_samples = (
            coords[:, 1].repeat_interleave(ray_n_samples).reshape(-1, ray_n_samples)
        )
        phi_samples = (
            coords[:, 2].repeat_interleave(ray_n_samples).reshape(-1, ray_n_samples)
        )

        # Note: r_samples is of size n_selected_px*arc_n_samples x ray_n_samples
        # so each row of r_samples contain r values for points picked from the same ray (should have the same theta and phi values)
        # theta_samples is also of size  n_selected_px*arc_n_samples x ray_n_samples
        # since all arc_n_samples x ray_n_samples  have the same value of theta, then the first n_selected_px rows have all the same value
        # Finally phi_samples is  also of size  n_selected_px*arc_n_samples x ray_n_samples
        # but not each ray has a different phi value

        # pts contain all points and is of size n_selected_px*arc_n_samples*ray_n_samples, 3
        # the first ray_n_samples rows correspond to points along the same ray
        # the first ray_n_samples*arc_n_samples row correspond to points along rays along the same arc
        pts = torch.stack((r_samples, theta_samples, phi_samples), dim=-1).reshape(-1, 3)

        dists = torch.diff(r_samples, dim=1)
        dists = torch.cat(
            [dists, torch.Tensor([sonar_resolution]).expand(dists[..., :1].shape)], -1
        )

        # r_samples_mid = r_samples + dists/2

        X_r_rand = pts[:, 0] * torch.cos(pts[:, 1]) * torch.cos(pts[:, 2])
        Y_r_rand = pts[:, 0] * torch.sin(pts[:, 1]) * torch.cos(pts[:, 2])
        Z_r_rand = pts[:, 0] * torch.sin(pts[:, 2])
        pts_r_rand = torch.stack((X_r_rand, Y_r_rand, Z_r_rand, torch.ones_like(X_r_rand)))

        pts_r_rand = torch.matmul(c2w, pts_r_rand)

        pts_r_rand = torch.stack((pts_r_rand[0, :], pts_r_rand[1, :], pts_r_rand[2, :]))

        # Centering step
        pts_r_rand = pts_r_rand.T - cube_center
        return dirs, dphi, None, rs, pts_r_rand, dists
    else:
        rays_o = origin[:, :3].expand(dirs.shape)
        render_step_size = 0.05 # TODO: make this less hacky 
        t_max = r.repeat_interleave(arc_n_samples)
        ray_indices, t_starts, t_ends = estimator.sampling(rays_o, dirs, render_step_size=render_step_size, near_plane=r_min, t_max=t_max)
        pts_r_rand = rays_o[ray_indices] + t_starts[:, None] * dirs[ray_indices]
        dists = t_ends - t_starts 

        # handle endpoints in a hacky way 
        dists_ep = torch.tensor(sonar_resolution).expand(t_max.shape) 
        pts_ep = rays_o + t_max[:, None] * dirs 
        pts_r_rand = torch.cat([pts_r_rand, pts_ep], dim=0)
        dists = torch.cat([dists, dists_ep], dim=0)
        # print(pts_r_rand.shape, ray_indices.shape)

        return dirs, dphi, ray_indices, None, pts_r_rand, dists


def select_coordinates(coords_all, target, N_rand, select_valid_px):
    if select_valid_px:
        coords = torch.nonzero(target)
    else:
        select_inds = torch.randperm(coords_all.shape[0])[:N_rand]
        coords = coords_all[select_inds]
    return coords
