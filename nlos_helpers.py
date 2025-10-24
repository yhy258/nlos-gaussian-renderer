import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
# import matplotlib
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.stats import multivariate_normal
# import scipy.io
# import pyvista as pv
# pv.set_jupyter_backend('trame')
from skimage.measure import marching_cubes
import open3d as o3d
import scipy
# from scipy.interpolate import griddata
from gaussian_model.gaussian_model import GaussianModel

# Try to import CUDA renderer
try:
    from gaussian_model.rendering_cuda import create_cuda_renderer, CUDA_AVAILABLE
    CUDA_RENDERER = create_cuda_renderer()
except ImportError:
    CUDA_AVAILABLE = False
    CUDA_RENDERER = None
    print("Note: CUDA renderer not available. Using standard implementation.")


def save_model(args, model, current_iter):
    # save model
    model_save_rel_dir = args.model_save_rel_dir
    model_dir = model_save_rel_dir
    os.makedirs(model_dir, exist_ok=True)
    model_name = f'{model_dir}/current_iter' + str(current_iter) + '.pt'
    params = model.get_params()
    torch.save(params, model_name)
    return 0

def gaussian2volume(args, model: GaussianModel, data_kwargs, camera_pos, resolution=128, mode='voxel'):
    with torch.no_grad():
        input_points, I1, I2, num_r, dtheta, dphi, theta_min, theta_max, phi_min, phi_max = spherical_sample_histogram(args, data_kwargs, camera_pos)
        input_points_ori = input_points[:, 0:3] # spatial coordinate
        result, density, albedo = model.estimate_rho_w(input_points_ori, camera_pos, c=data_kwargs['c'], deltaT=data_kwargs['deltaT'], scaling_modifier=args.scaling_modifier, out_separately=True)
        irregular_density = density.cpu().numpy()
        irregular_albedo = albedo.cpu().numpy()
    irregular_points = input_points_ori.cpu().numpy()


    if mode.lower() == 'mesh':
        threshold = np.mean(irregular_density)
        dense_points = irregular_points[irregular_density > threshold]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(dense_points)

        # estimate orthogonal vector
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

        # Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8)

        # 5. Remove triangles that have low density
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)

        # 6. result save.
        o3d.io.write_point_cloud("output_point_cloud.ply", pcd)
        o3d.io.write_triangle_mesh("output_poisson_mesh.ply", mesh)



def cartesian2spherical(pt): # (x, y, z) -> (r, \theta, \phi)
    # cartesian to spherical coordinates
    # input： pt N x 3 ndarray

    spherical_pt = np.zeros(pt.shape)
    spherical_pt[:,0] = np.sqrt(np.sum(pt ** 2,axis=1))
    spherical_pt[:,1] = np.arccos(pt[:,2] / spherical_pt[:,0])
    phi_yplus = (np.arctan(pt[:,1] / (pt[:,0] + 1e-8))) * (pt[:,1] >= 0)
    phi_yplus = phi_yplus + (phi_yplus < 0).astype(np.int32) * (np.pi)
    phi_yminus = (np.arctan(pt[:,1] / (pt[:,0] + 1e-8))) * (pt[:,1] < 0)
    phi_yminus = phi_yminus + (phi_yminus > 0).astype(np.int32) * (-np.pi)
    spherical_pt[:,2] = phi_yminus + phi_yplus
    return spherical_pt

def cartesian2spherical_torch(pt):  # (x, y, z) -> (r, \theta, \phi)
    # cartesian to spherical coordinates
    # input： pt N x 3 torch.Tensor
    spherical_pt = torch.zeros(pt.shape, device=pt.device)
    r = torch.linalg.norm(pt, dim=1)
    spherical_pt[:,0] = r
    spherical_pt[:,1] = torch.acos(pt[:, 2] / r)
    spherical_pt[:,2] = torch.atan2(pt[:, 1], pt[:, 0])
    return spherical_pt


def spherical2cartesian_torch(pt): # N x 3 -> N x 3. (r, \theta, \phi) -> (x, y, z)
    cartesian_pt = torch.zeros(pt.shape, device=pt.device)
    cartesian_pt[:,0] = pt[:,0] * torch.sin(pt[:,1]) * torch.cos(pt[:,2])
    cartesian_pt[:,1] = pt[:,0] * torch.sin(pt[:,1]) * torch.sin(pt[:,2])
    cartesian_pt[:,2] = pt[:,0] * torch.cos(pt[:,1])

    return cartesian_pt


def volume_box_point(volume_position, volume_size):
    """
    args
        volume_position: The center position of the hidden volume: (3,)
        volume_size: The size of the hidden volume: Scalar
    """
    xv, yv, zv = volume_position # center position
    x = np.array([xv - volume_size / 2, xv - volume_size / 2, xv - volume_size / 2, xv - volume_size / 2, xv + volume_size / 2, xv + volume_size / 2, xv + volume_size / 2, xv + volume_size / 2])
    y = np.array([yv - volume_size / 2, yv - volume_size / 2, yv + volume_size / 2, yv + volume_size / 2, yv - volume_size / 2, yv - volume_size / 2, yv + volume_size / 2, yv + volume_size / 2])
    z = np.array([zv - volume_size / 2, zv + volume_size / 2, zv - volume_size / 2, zv + volume_size / 2, zv - volume_size / 2, zv + volume_size / 2, zv - volume_size / 2, zv + volume_size / 2])
    box = np.stack((x, y, z),axis = 1)
    return box # output: 8 by 3. The vertex of the volume cube.



# def spherical_sample_histogram(camera_grid_positions, volume_position, volume_size, deltaT, c,  num_sampling_points, start, end):
# make torch functions.
def spherical_sample_histogram(args, data_kwargs, current_camera_grid_positions):
    """
    args
        data_kwargs:
            index: the indices for the shuffled data
            camera_grid_positions: The position of the camera grid (visible wall?); (Na x 3) : torch.Tensor
            camera_grid_size: The size of the camera grid : scalar
            volume_position: The center position of the hidden volume: (3,): torch.Tensor
            volume_size: The size of the hidden volume: Scalar
            volume_box_point: The vertex of the volume cube.
            deltaT: The discrete time interval in this setting
            c: The speed of the light
            pmin and pmax: the range of the volume coordinate
    """


    ### The distance unit is µm.
    ### and the data_kwargs' parameters have actual distance values except for "index" parameter
    x0, y0, z0 = current_camera_grid_positions
    assert isinstance(current_camera_grid_positions, torch.Tensor), "current camera grid position parameter should be torch.Tensor"

    box_point = data_kwargs['volume_box_point']
    device = box_point.device

    # shift the origin of the volume to the camera from the world's origin
    box_point = box_point - current_camera_grid_positions[None, :] # 8 by 3
    # cartesian 2 spherical coordinate system
    sphere_box_point = cartesian2spherical_torch(box_point) # 8, 3 (r, \theta, \phi)
    # set the angular bound of the volume
    theta_min = torch.min(sphere_box_point[:, 1]).item()
    theta_max = torch.max(sphere_box_point[:, 1]).item()
    phi_min = torch.min(sphere_box_point[:, 2]).item()
    phi_max = torch.max(sphere_box_point[:, 2]).item()

    # make angular grid
    num_sampling_points = args.num_sampling_points
    theta = torch.linspace(theta_min, theta_max, num_sampling_points, dtype=torch.float, device=device)
    phi = torch.linspace(phi_min, phi_max, num_sampling_points, dtype=torch.float, device=device)

    dtheta = (theta_max - theta_min) / num_sampling_points
    dphi = (phi_max - phi_min) / num_sampling_points

    # make radius grid (ray distance)
    c = data_kwargs['c']
    deltaT = data_kwargs['deltaT']
    r_min = args.start * c * deltaT
    r_max = args.end * c * deltaT
    num_r = args.end - args.start
    r = torch.linspace(r_min, r_max, num_r, dtype=torch.float, device=device) # Nr

    I1 = math.floor(r_min / (c * deltaT)) # start idx
    I2 = math.ceil(r_max / (c * deltaT)) # end idx
    num_r = r.shape[0] # The number of samples.

    grid = torch.stack(torch.meshgrid(r, theta, phi), axis=-1) # Nr, Ns, Ns, 3. This tensor would be already in GPU

    spherical = grid.reshape([-1,3]) # (N, 3) where N = NrNs^2
    cartesian = spherical2cartesian_torch(spherical)
    cartesian = cartesian + current_camera_grid_positions # re-shift the center of the volume from the cam space to the world space

    # (x, y, z; theta, phi) positions for the evaluation.
    # x, y, z are the absolute position
    # theta and phi are the relative parameters for the given camera position
    cartesian = torch.cat((cartesian, spherical[:,1:3]), axis = 1).float() # x, y, z, theta, phi
    return cartesian, I1, I2, num_r, dtheta, dphi, theta_min, theta_max, phi_min, phi_max



def gaussian_transient_rendering(args, model, data_kwargs, input_points, current_camera_grid_positions, I1, I2, num_r, dtheta, dphi):
    """
    Gaussian transient rendering with optional CUDA acceleration.
    
    If args.use_cuda_renderer is True and CUDA renderer is available,
    uses efficient ray-based CUDA kernels. Otherwise falls back to standard implementation.
    """
    # Check if CUDA rendering is requested and available
    if hasattr(args, 'use_cuda_renderer') and args.use_cuda_renderer and CUDA_RENDERER is not None:
        return gaussian_transient_rendering_cuda(
            args, model, data_kwargs, input_points, 
            current_camera_grid_positions, I1, I2, num_r, dtheta, dphi
        )
    
    # Standard implementation
    # Result: Na (Na = Nr x Ntheta x Nphi)
    input_points_ori = input_points[:, 0:3] # spatial coordinate Na by 3
    if args.occlusion == False:
        result = model.estimate_rho_w_no_occlusion(input_points_ori, current_camera_grid_positions, c=data_kwargs['c'], deltaT=data_kwargs['deltaT'], scaling_modifier=args.scaling_modifier)
    else:
        result = model.estimate_rho_w(input_points_ori, current_camera_grid_positions, c=data_kwargs['c'], deltaT=data_kwargs['deltaT'], scaling_modifier=args.scaling_modifier)
    ### Attenuation (Confocal Setting)
    # print("The shape of the result: ", result.shape)

    result = result.reshape(num_r, args.num_sampling_points ** 2)

    with torch.no_grad():
        distance = (torch.linspace(I1, I2, num_r, dtype=torch.float, device=input_points.device) * data_kwargs['deltaT'] * data_kwargs['c']) # I1, I2 are the start and end index. I1 * deltaT * c and I2 * deltaT * c would be the real start and end distance of the ray.
        # num_r is the number of samples for radius (ray).
        distance = distance.view(-1, 1)
        distance = distance.repeat(1, args.num_sampling_points ** 2)
        Theta = input_points.view(-1, args.num_sampling_points ** 2, 5)[:, :, 3] # (Nr, Na^2, 5)

    result = result / (distance ** 2) * torch.sin(Theta) # attenuation factor. sin(theta) / r^2
    result = result * (data_kwargs['volume_position'][1] ** 2) # WHAT?? WHY?

    pred_histogram = torch.sum(result, axis=1) # summation for the angular components.
    pred_histogram = pred_histogram * dtheta * dphi # infinitesimal factor product.
    # print("Predicted histogram's shape: ", pred_histogram.shape)

    return result, pred_histogram


def gaussian_transient_rendering_cuda(args, model, data_kwargs, input_points, current_camera_grid_positions, I1, I2, num_r, dtheta, dphi):
    """
    CUDA-accelerated version of gaussian_transient_rendering.
    
    Uses ray-based rendering with Gaussian filtering for efficient computation.
    Only relevant Gaussians are processed for each ray.
    """
    # Extract angular ranges from input_points
    theta_vals = input_points[:, 3]
    phi_vals = input_points[:, 4]
    
    theta_min = theta_vals.min().item()
    theta_max = theta_vals.max().item()
    phi_min = phi_vals.min().item()
    phi_max = phi_vals.max().item()
    
    r_min = I1 * data_kwargs['c'] * data_kwargs['deltaT']
    r_max = I2 * data_kwargs['c'] * data_kwargs['deltaT']
    
    # Call CUDA renderer
    result_3d, pred_histogram = CUDA_RENDERER.render_transient(
        gaussian_model=model,
        camera_pos=current_camera_grid_positions,
        theta_range=(theta_min, theta_max),
        phi_range=(phi_min, phi_max),
        r_range=(r_min, r_max),
        num_theta=args.num_sampling_points,
        num_phi=args.num_sampling_points,
        num_r=num_r,
        c=data_kwargs['c'],
        deltaT=data_kwargs['deltaT'],
        scaling_modifier=args.scaling_modifier,
        use_occlusion=args.occlusion,
        rendering_type=args.rendering_type
    )
    
    # Reshape to match original format [num_r, num_angular^2]
    result = result_3d.reshape(num_r, args.num_sampling_points * args.num_sampling_points)
    
    # Apply the mysterious scaling factor (kept for compatibility)
    result = result * (data_kwargs['volume_position'][1] ** 2)
    pred_histogram = pred_histogram * (data_kwargs['volume_position'][1] ** 2)
    
    return result, pred_histogram

def compute_loss(args, model: GaussianModel, data_kwargs: dict, optim_kwargs: dict, device: torch.device):
    """
        data_kwargs:
            index: the indices for the shuffled data
            camera_grid_positions: The position of the camera grid (visible wall?); (3 x Na) : torch.Tensor
            camera_grid_size: The size of the camera grid : scalar
            volume_position: The center position of the hidden volume: (3,): torch.Tensor
            volume_size: The size of the hidden volume: Scalar
            volume_box_point: The vertex of the volume cube.
            deltaT: The discrete time interval in this setting
            c: The speed of the light
            pmin and pmax: the range of the volume coordinate
        optim_kwargs:
            prev_time: The start time of the training
            M, m, N, n : indices for the current step
            where N is the number of columns, m is the current index of rows, and n is the current index of columns.
            (j: target histogram. According to the given setting, we can accumulate the loss using the indices j
            N_iters: epochs
            criterion: the main loss function of the model
            # optimizer is in the GaussianModel instance.
    """
    ## get the current virtual camera position
    m, N, n = optim_kwargs['m'], optim_kwargs['N'], optim_kwargs['n']
    v_flatten_pos = m * N + n

    camera_grid_positions = data_kwargs['camera_grid_positions']
    current_camera_grid_positions = camera_grid_positions[:, v_flatten_pos]

    ## get volume position bound
    pmin, pmax = data_kwargs['pmin'], data_kwargs['pmax']

    with torch.no_grad():
        ### only consider the confocal setting
        # input_points: (N, 3) where N = Nr*Na^2. Nr is the number of samples for radius, and Na is the number of samples for angular components (theta and phi.)
        # and Na == args.num_sampling_points
        input_points, I1, I2, num_r, dtheta, dphi, theta_min, theta_max, phi_min, phi_max = spherical_sample_histogram(args, data_kwargs, current_camera_grid_positions)

    #### We should devise the following function
    # print("The number of r's sampling points", num_r)
    result, pred_histogram  = gaussian_transient_rendering(args, model, data_kwargs, input_points, current_camera_grid_positions, I1, I2, num_r, dtheta, dphi)
    # print(f"Thue shape of the result: {result.shape} and the shape of the input points (sampling points): {input_points.shape}")
    #### Matching the time indices
    with torch.no_grad():
        nlos_histogram = data_kwargs['nlos_data'][I1:(I1 + num_r), m, n]
        nlos_histogram = nlos_histogram * args.gt_times
    loss = optim_kwargs['criterion'](pred_histogram, nlos_histogram)
    loss_coffe = torch.mean(nlos_histogram ** 2)
    equal_loss = loss / loss_coffe

    if args.save_fig:
        # i: epoch (not iteration.)
        if (optim_kwargs['current_iter'] % args.save_hist_fig_interval == 0):
            loss_show = equal_loss.cpu().detach().numpy()
            plt.plot(nlos_histogram.cpu(), alpha=0.5, label='data')
            plt.plot(pred_histogram.cpu().detach().numpy(), alpha = 0.5, label='predicted')
            # plt.plot(pred_histogram_extra.cpu().detach().numpy(), alpha = 0.5, label='predicted extra')
            plt.legend(loc='upper right')
            # plt.title('grid position:' + str(x0) + ' ' + str(z0))
            plt.title('grid position:' + str(format(current_camera_grid_positions[0].item(), '.4f')) + ' ' + str(format(current_camera_grid_positions[2].item(), '.4f')) + ' equal loss:' + str(format(loss_show, '.8f')) + ' coffe:' + str(format(loss_coffe.cpu().detach().numpy(), '.8f')))
            os.makedirs(f'./figure/', exist_ok=True)
            plt.savefig(f'./figure/' + str(optim_kwargs['current_iter']) + '_' + str(m) + '_' + str(n))
            plt.close()

    mdic = {'nlos':nlos_histogram.cpu().detach().numpy(),'pred':pred_histogram.cpu().detach().numpy()}
    scipy.io.savemat('./loss_compare.mat', mdic)

    return loss, equal_loss

def batch_compute_loss(args, model: GaussianModel, data_kwargs: dict, optim_kwargs: dict, device: torch.device):
    camera_grid_positions = data_kwargs['camera_grid_positions'] # 3 x Na
    pmin, pmax = data_kwargs['pmin'], data_kwargs['pmax']
    # with torch.no_grad():
