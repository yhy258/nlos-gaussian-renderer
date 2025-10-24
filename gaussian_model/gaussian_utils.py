import numpy as np
import torch
from tqdm import tqdm
import open3d as o3d
import trimesh


def init_rand_points(args, data_kwargs, margin=0.1, rho_scale=0.1, device='cuda'):
    # if use angular info == 0, we can conduct biased sampling toward the angular range.
    """
        pmin: (3,) [spatial, angular (phi, rho)]
        pmax: (3,) [spatial, angular (phi, rho)]
    """
    # initial gaussian num
    init_gaussian_num = args.init_gaussian_num
    # sampling rho
    rho = np.random.rand(init_gaussian_num, 1) * rho_scale
    # Sampling
    pmin, pmax = data_kwargs['pmin'], data_kwargs['pmax']
    pmin_cart, pmax_cart = pmin[:3].cpu().numpy(), pmax[:3].cpu().numpy() # 3,

    # samples = torch.rand((init_gaussian_num, 3), device=device)
    samples = np.random.rand(init_gaussian_num, 3)
    # rho = torch.rand((init_gaussian_num, 1), device=device)


    ### initialization: Avoid outside points
    modified_pmin_x = pmin_cart + np.abs(pmin_cart*margin)
    modified_pmax_x = pmax_cart - np.abs(pmax_cart*margin)
    samples = samples * (modified_pmax_x[None] - modified_pmin_x[None]) + modified_pmin_x[None] # N, 3

    return samples, rho


"""
    Employ : space carving.
"""
def detect_first_bounces(transient, threshold=1e-5):
    bins, height, width = transient.shape


    first_bounces = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            if np.sum(transient[:,y,x]) != 0:
                for b in range(1, bins, 1):
                    if transient[b,y,x] - transient[b-1,y,x]  > threshold:
                        first_bounces[y,x] = b
                        break
    return first_bounces

# This code is based on https://github.com/yfujimura/nlos-neus/blob/master/space_carving.py
def space_carving(args, data_kwargs):
    """
        data_kwargs: (Except for scalar values, the whole data is torch.Tensor)
            nlos_data:
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
    if args.scene == "zaragoza_bunny":
        nlos_file = "data/zaragozadataset/zaragoza256_preprocessed.mat"
        dataset_type = "zaragoza256"
        start = 0
        threshold = 1e-5 # we only consider this.

    camera_grid_positions = data_kwargs['camera_grid_positions']
    volume_position = data_kwargs['volume_position']
    volume_size = data_kwargs['volume_size']
    nlos_data = data_kwargs['nlos_data'].cpu().numpy()
    c, deltaT = data_kwargs['c'], data_kwargs['deltaT']

    ### shift the origin.
    camera_grid_positions = camera_grid_positions - volume_position[:, None]
    volume_position_np = np.zeros((1, 3)) # use numpy lib.
    vmin = volume_position - volume_size / 2
    vmax = volume_position + volume_size / 2

    radiuses = start + detect_first_bounces(nlos_data[start:], threshold=threshold)
    # L, M, N = nlos_data.shape
    radiuses = radiuses * c * deltaT
    radiuses = radiuses.reshape(-1,) # LMN

    unit_distance = volume_size / (args.carving_volume_size-1)
    xv = yv = zv = np.linspace(-volume_size / 2, volume_size / 2, args.carving_volume_size)

    coords = np.stack(np.meshgrid(xv, yv, zv, indexing='ij'),-1) # coords
    # coords = coords.transpose([1,0,2,3]) (If i use indexing='xy')
    coords = coords.reshape([-1,3]) # NT, 3
    coords = torch.from_numpy(coords.astype(np.float32)).to(camera_grid_positions.device)

    votes = torch.zeros(coords.shape[0])

    print("space carving...")

    total_votes = 0
    with torch.no_grad():
        for i in tqdm(range(0, camera_grid_positions.shape[1], 1)):
            if radiuses[i] > 0:
                total_votes += 1

                pt0 = camera_grid_positions[:,i]

                v = coords - pt0[None,:]
                diffs = torch.norm(v, dim=1)
                mask = torch.ones_like(diffs)
                mask[diffs < radiuses[i]] = 0
                votes[mask > 0] = votes[mask > 0] + 1

    threshold = torch.max(votes).item()*args.space_carving_ratio
    mask = torch.zeros_like(votes)
    mask[votes > threshold] = 1
    mask[votes <= threshold] = 0


    print("Complete voting.")
    # coords: NT, 3 -> Nt, 3
    # coords + volumepos : Nt, 3
    # print(volume_position.device, camera_grid_positions.device)

    # print(coords.device, mask.device, volume_position.device)
    coords2 = coords[torch.nonzero(mask, as_tuple=True)[0]] + volume_position[None] # volume_position: 1 x 3
    return coords2

def sample_from_feasible_space_jittering(args, data_kwargs, margin=0.1, rho_scale=0.1, device='cuda', exact_mesh_samping=False):
    # if use angular info == 0, we can conduct biased sampling toward the angular range.
    """
        pmin: (3,) [spatial, angular (phi, rho)]
        pmax: (3,) [spatial, angular (phi, rho)]
    """

    # initial gaussian num
    init_gaussian_num = args.init_gaussian_num
    # sampling rho
    rho = np.random.rand(init_gaussian_num, 1) * rho_scale

    # point sampling
    coords2 = space_carving(args, data_kwargs) # coords2: torch.Tensor. Nt, 3

    if exact_mesh_samping:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords2.cpu().numpy())
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        samples, _ = trimesh.sample.sample_surface(mesh_trimesh, count=init_gaussian_num)
    else:
        pmin, pmax = data_kwargs['pmin'], data_kwargs['pmax']
        spacing = (pmax - pmin) / (args.carving_volume_size-1)
        spacing = spacing[:3]
        half_spacing = spacing / 2.0
        random_indices = torch.randint(0, coords2.shape[0], (init_gaussian_num,), device=coords2.device)
        base_points = coords2[random_indices] # Ng x 3

        random_offsets = (torch.rand_like(base_points) - 0.5) * 2 * half_spacing[None]
        samples = base_points + random_offsets

    return samples, rho


## misc funcs for Gaussian Model

def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper