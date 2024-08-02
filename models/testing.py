import torch
import torch.nn as nn
import plotly.graph_objects as go
import math
import numpy as np

def plot_sphere(r, fig):
    azis = torch.linspace(-np.pi, np.pi, 64)
    eles = torch.linspace(-np.pi / 2, np.pi / 2, 64)
    pixels_theta, pixels_phi = torch.meshgrid(
        azis, eles, indexing="ij"
    )  # careful with indexing here

    xs = r * torch.cos(pixels_theta) * torch.cos(pixels_phi)
    ys = r * torch.sin(pixels_theta) * torch.cos(pixels_phi)
    zs = r * torch.sin(pixels_phi)
    p = torch.stack([xs, ys, zs], dim=-1)  # 64, 64, 3

    sphere_surface = go.Surface(
        z=p[:, :, 2].cpu(),
        x=p[:, :, 0].cpu(),
        y=p[:, :, 1].cpu(),
        opacity=0.3,
        showscale=False,
    )
    fig.add_trace(sphere_surface)

def convert_pose(pose, direction):
    """
    Args:
        pose: (4, 4)
    Returns:
        converted_pose: (4, 4)

    neus: (right, down, in)
    ho: (in, left, up)
    ue: (in, right, up)

    how to visualize: if pose is in coord system A, then to convert to coord system
        B c_mat needs to convert coords from coord system B to coord system A. Think
        of coord system A vec first, where do you get the things you need from a vec
        written wrt coord system B?
    """
    conversion_mats = {
        "neus_to_ho": np.array([[0.0, -1.0, 0.0], 
                                [0.0, 0.0, -1.0], 
                                [1.0, 0.0, 0.0]]),
        "ue_to_neus": np.array(
            [[0.0, 0.0, 1.0], 
             [1.0, 0.0, 0.0], 
             [0.0, -1.0, 0.0]]
        ),  # sign on last row?
    }
    c_mat = conversion_mats[direction]
    if isinstance(pose, np.ndarray):
        converted_pose = pose.copy()
    elif isinstance(pose, torch.Tensor):
        convert_pose = pose.clone()
    converted_pose[:3, :3] = c_mat.T @ pose[:3, :3] @ c_mat
    # converted_pose[:3, :3] = pose[:3, :3] @ c_mat
    # converted_pose[:3, :3] = c_mat.T @ pose[:3, :3]
    converted_pose[:3, 3:] = c_mat.T @ pose[:3, 3:]
    return converted_pose

def plot_box(r, fig):
    range = torch.linspace(-r, r, 10)
    r1, r2 = torch.meshgrid(range, range)
    lower = torch.ones_like(r1) * (-r)
    upper = torch.ones_like(r1) * r
    planes = [
        [r1, r2, lower],
        [r1, r2, upper],
        [lower, r1, r2],
        [upper, r1, r2],
        [r1, lower, r2],
        [r1, upper, r2],
    ]
    for p in planes:
        p1 = torch.stack(p, dim=-1)
        p1_s = go.Surface(
            z=p1[:, :, 2].cpu(),
            x=p1[:, :, 0].cpu(),
            y=p1[:, :, 1].cpu(),
            opacity=0.3,
            showscale=False,
        )
        fig.add_trace(p1_s)


# can refactor below slightly, curry parameter into sdf_func
class ShapeSDF(nn.Module):
    def __init__(self, sdf_func, parameter):
        super().__init__()
        self.parameter = parameter
        self.sdf_func = sdf_func

    def forward(self, inputs):
        """
        Args:
            inputs: batch_size X 3
        """
        dummy_feats = torch.zeros((inputs.size()[0], 1)).cuda()
        sdf_vals = self.sdf_func(inputs, self.parameter)
        return torch.cat([sdf_vals, dummy_feats], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def gradient(self, x, mode="analytic"):
        if mode == "analytic":
            x.requires_grad_(True)
            y = self.sdf(x)
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0].unsqueeze(1)
        elif mode == "finite_difference":
            gradients = finite_difference_grad(self.sdf, x)
        return gradients
    

def sphere_sdf_func(inputs, radius):
    norms = torch.linalg.vector_norm(inputs, dim=-1, keepdim=True)
    sdf_vals = norms - radius
    return sdf_vals


def box_sdf_func(inputs, bounds):
    # print(inputs.size(), bounds.size())
    # from inigo quilez's site
    q = torch.abs(inputs) - bounds
    norms = torch.linalg.vector_norm(torch.clamp(q, min=0.0), dim=-1, keepdim=True)
    other = torch.min(
        torch.max(q[:, 0:1], torch.max(q[:, 1:2], q[:, 2:3])),
        torch.zeros_like(q[:, 0:1]),
    )
    return norms + other

# color based on normals? lambertian reflectance?
class RenderNetLamb(nn.Module):
    def __init__(self, sdf_func=None, falloff=-10, channels=1):
        super().__init__()
        self.sdf_func = sdf_func
        self.falloff = falloff
        self.channels = channels

    def forward(self, points, normals, view_dirs, feature_vectors):
        # print(points.shape)
        cos = (normals * view_dirs).sum(-1)[..., None].repeat(1, self.channels).abs()
        if self.sdf_func is not None:
            sdf_vals = self.sdf_func(points)
            weights = torch.exp(
                self.falloff * torch.abs(sdf_vals.expand((-1, self.channels)))
            )
            res = cos * weights
        else:
            res = cos
        return res


class ConstVar(nn.Module):
    def __init__(self, inv_var):
        super().__init__()
        self.variance = inv_var

    def forward(self, pts):
        return torch.tensor(
            [[math.exp(10 * self.variance)]]
        ).cuda()  # previously had sign error here, be careful
    
def plot_mesh(mesh, fig):
    """
    Args:
        mesh: trimesh.Trimesh 
        fig: graph_objects.Figure
    """
    mesh_plot = go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        intensity = np.linspace(0, 1, len(mesh.faces), endpoint=True),
        intensitymode='cell',
    )
    fig.add_trace(mesh_plot)

def plot_points_3d(points, fig, size=2, mode="markers"):
    """
    Args:
        points: (n, 3) np.ndarray
        fig: graph_objects.Figure
    """
    center_plot = go.Scatter3d(x=points[:, 0],
                            y=points[:, 1],
                            z=points[:, 2],
                            mode=mode,
                            marker=dict(size=size))
    fig.add_trace(center_plot)

def sample_points_along_rays(rays_o, rays_d, near, far, num_points=16):
    """
    Args: 
        rays_o: (n, 3), np.ndarray
        rays_d: (n, 3), np.ndarray
        near: float 
        far: float
    Returns:
        vis_sample_pts: (n, num_points, 3)
    """
    vis_r = np.linspace(near, far, num_points) # (num_points,)
    vis_sample_pts = rays_o[:, None, :] + rays_d[:, None, :] * vis_r[None, :, None]
    # vis_sample_pts = vis_sample_pts.reshape(-1, 3)
    return vis_sample_pts

def plot_pose_axes(pose, fig):
    """
    Args:
        pose: (4, 4) np.ndarray
        fig: graph_objects.Figure
    """
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().numpy()
    t = pose[:3, 3].T
    colorscale = [
        [0, "red"],
        [1.0, "green"],
    ]  # red is origin, green is point inwards wrt ray direction
    for i in range(3):
        curr_in = pose[:3, i]# z-axis is in
        far = t + 0.5 * curr_in
        endpoints = np.stack([t, far])
        dir_vec_plot = go.Scatter3d(x=endpoints[:, 0], y=endpoints[:, 1], z=endpoints[:, 2],
                                marker=dict(size=3, color=[0.0, 1.0], colorscale=colorscale),
                                line=dict(color="darkblue", width=2))
        fig.add_trace(dir_vec_plot) 
        