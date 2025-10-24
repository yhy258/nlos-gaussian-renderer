import numpy as np
import torch
import torch.nn as nn
from .gaussian_utils import build_rotation,build_scaling_rotation, strip_symmetric, inverse_sigmoid, inverse_opacity_activation, get_expon_lr_func
from .sh_utils import eval_sh, RHO2SH

try:
    from simple_knn._C import distCUDA2 ### KNN using CUDA.
    KNN_FLAG = True
except:
    KNN_FLAG = False

class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize



    def __init__(self, args, device):
        self.args = args
        ### Albedo parameters (Spherical Harmonics or Constant)
        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree # if we set the max_sh_degree == 0, this would be equal to the constant albedo.
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)

        ### A position parameter
        self._mu = torch.empty(0)

        ### Covariance parameters
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)

        ### Opacity of the Gaussian
        self._opacity = torch.empty(0)

        ### Non-trainable.
        self.mu_grad_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.spatial_lr_scale = 0

        ### Optimizer
        self.optimizer = None

        self.setup_functions()
        self.device = device


    def get_params(self):
        return {
            'mu': self._mu,
            'features_dc': self._features_dc,
            'features_rest': self._features_rest,
            'opacity': self._opacity,
            'scaling': self._scaling,
            'rotation': self._rotation,
            'optimizer': self.optimizer,
            'max_sh_degree': self.max_sh_degree,
            'active_sh_degree': self.active_sh_degree
        }

    def restore(self, load_path, training_args):
        print(f"Load Gaussian parameters from '{load_path}' ...")
        params = torch.load(load_path, map_location=self.device, weights_only=False)

        # 1. Learnable parameters
        self._mu = nn.Parameter(params['mu'].requires_grad_(True))
        self._features_dc = nn.Parameter(params['features_dc'].requires_grad_(True))
        self._features_rest = nn.Parameter(params['features_rest'].requires_grad_(True))
        self._opacity = nn.Parameter(params['opacity'].requires_grad_(True))
        self._scaling = nn.Parameter(params['scaling'].requires_grad_(True))
        self._rotation = nn.Parameter(params['rotation'].requires_grad_(True))

        # 2. SH degrees
        self.max_sh_degree = params['max_sh_degree']
        self.active_sh_degree = params['active_sh_degree']

        # 3. Optimizer
        if self.optimizer is None:
            # training_setup 등을 통해 옵티마이저를 먼저 생성해야 함
            print("Warning: Optimizer is not initialized. Let's initialize it ")
            self.training_setup(training_args)
        else:
            if isinstance(params['optimizer'], torch.optim.Adam):
                self.optimizer.load_state_dict(params['optimizer'].state_dict())
            else:
                self.optimizer.load_state_dict(params['optimizer'])

        print("Restoration complete")



    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_mu(self):
        return self._mu

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_bboxes(self, scaling_modifier=1.0, sigma_scale=3.0):
        """
        AABB

        Args:
            scaling_modifier (float):
            sigma_scale (float): sigma scale that determines BBox size (일반적으로 3.0).

        Returns:
            torch.Tensor: (N, 2, 3) size tensor.
                          [:, 0, :] is bbox_min [x, y, z]
                          [:, 1, :] is bbox_max [x, y, z]
        """
        # 1. Get the center of the Gaussians
        mu = self.get_mu  # (N, 3)

        # 2. Diagonal factors of the Covariance matrices.
        scaling = self.get_scaling
        rotation = self.get_rotation

        # L = R @ S
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)

        # Cov = L @ L.T
        actual_covariance = L @ L.transpose(1, 2)  # (N, 3, 3)

        # 3. AABB's half of extents
        # AABB extent = sigma_scale * sqrt(diag(Cov))
        diag_elements = torch.diagonal(actual_covariance, dim1=-2, dim2=-1)  # (N, 3)
        extents = sigma_scale * torch.sqrt(torch.clamp_min(diag_elements, 1e-8))  # (N, 3)

        # 4. AABB's min max coords
        bbox_min = mu - extents  # (N, 3)
        bbox_max = mu + extents  # (N, 3)

        # 5. (N, 2, 3) STACK!
        bboxes = torch.stack([bbox_min, bbox_max], dim=1)

        return bboxes

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_params(self, points, rho, pmin, pmax, spatial_lr_scale=1.0): # rho : albedo
        self.spatial_lr_scale = spatial_lr_scale

        if isinstance(points, torch.Tensor):
            points = points.cpu()
        if isinstance(rho, torch.Tensor):
            rho = rho.cpu()
        fused_point_cloud = torch.tensor(np.asarray(points), dtype=torch.float, device=self.device)
        fused_rho = RHO2SH(torch.tensor(np.asarray(rho), dtype=torch.float, device=self.device))
        #### albedo would be represented as Spherical harmonics

        features = torch.zeros((fused_rho.shape[0], 1, (self.max_sh_degree + 1) ** 2), dtype=torch.float, device=self.device)
        features[:, :1, 0 ] = fused_rho
        features[:, 1:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # Initialize covariances using KNN (I guess near the three points?
        if KNN_FLAG:
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(points), dtype=torch.float, device=self.device)), 0.0000001)
        else:
            pmin_x, pmax_x = pmin[0], pmax[0]
            init_gaussian_num = points.shape[0]
            dist2 = (pmax_x - pmin_x) / (init_gaussian_num + 1e-9)
            dist2 = torch.clamp_min(torch.tensor(dist2, dtype=torch.float, device=self.device), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), dtype=torch.float, device=self.device)
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device))

        # Initialize parameters
        self._mu = nn.Parameter(fused_point_cloud.requires_grad_(True)) # N x 3
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) # N x 1 x K
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True)) # N by 3
        self._rotation = nn.Parameter(rots.requires_grad_(True)) # N by 4
        self._opacity = nn.Parameter(opacities.requires_grad_(True)) # N by 1

    def training_setup(self, training_args):
        self.mu_grad_accum = torch.zeros((self.get_mu.shape[0], 1), dtype=torch.float, device=self.device)
        self.denom = torch.zeros((self.get_mu.shape[0], 1), dtype=torch.float, device=self.device)

        ### Define Param-Dict
        #### We should define each parameters.
        l = [
            {'params': [self._mu], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "mu"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.mu_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "mu":
                lr = self.mu_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr



    def estimate_gaussian_pdf(self, input_points_ori, scaling_modifier=1.):
        """
        Utilized params:
            input_points_ori : absolute position [x, y, z] (Na x 3)
            self._mu : the (learnable) mean params of the gaussians (Ng x 3)
            self._scaling & self._rotation : the (learnable) cov params of the gaussians

        Output:
            Estimated Gaussian PDF
            G(x; mu, cov): (Ng x Na)
        """
        # get Quaternion
        scales = self.scaling_activation(self.get_scaling*scaling_modifier)
        rotations = self.rotation_activation(self._rotation)
        # get Mean
        mu = self.get_mu

        Ng = mu.shape[0]
        Na = input_points_ori.shape[0]

        # Expand the dimension for broadcasting
        # diff's shape: (Ng, Na, 3)
        diff = input_points_ori.unsqueeze(0) - mu.unsqueeze(1)

        # 2. Mahalanobis distance
        # Inverse Cov = RS^{-2}R where S is the diagonal scaling matrix.
        # print(f"Rotation shape: {rotations.shape} and Diff shape: {diff.shape}")
        # Rotation shape: torch.Size([100, 4]) and Diff shape: torch.Size([100, 51200, 3])
        rots = build_rotation(rotations)
        T = torch.matmul(rots.unsqueeze(1), diff.unsqueeze(-1)).squeeze(-1)

        mahalanobis_sq = torch.sum((T / scales.unsqueeze(1))**2, dim=-1) # shape: (Ng, Na)
        exponent = -0.5 * mahalanobis_sq

        # # 3. Determinant
        # # |RSS^TR^T| = |R||S||S||R|
        # cov_det = (scales[:, 0] * scales[:, 1] * scales[:, 2])**2 # Ng
        # norm_const = 1.0 / torch.sqrt((2 * torch.pi)**3 * cov_det + 1e-9) # Ng
        # pdf = norm_const.unsqueeze(1) * torch.exp(exponent) # Ng x Na
        pdf = torch.exp(exponent) # neglecting the normalization factor like 3DGS

        return pdf


    def estimate_rho_w(self, input_points_ori, current_camera_grid_positions, c, deltaT, scaling_modifier=1.0, out_separately=False):
        # input_points_ori : absolute position [x, y, z] (Na x 3) where Na = Nr * N\theta * N\phi
        # current_camera_grid_positions: (3,)
        gaussian_pdf = self.estimate_gaussian_pdf(input_points_ori, scaling_modifier) # Ng x Na
        opacity = self.get_opacity # Ng x 1

        ### Calculate rho from SHs.
        #### View-dependent albedo
        shs_view = self.get_features.transpose(1, 2).view(-1, 1, (self.max_sh_degree+1)**2)
        dir_pp = self.get_mu - current_camera_grid_positions.unsqueeze(0) # Ng, 3
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)

        sh2rho = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
        rho = torch.clamp_min(sh2rho + 0.5, 0.0) # Ng by 1


        density = gaussian_pdf * opacity # Ng x Na
        num_r = self.args.end - self.args.start
        density = density.view(-1, num_r, self.args.num_sampling_points ** 2) # Ng x Nr x (Ns*Ns)
        if self.args.rendering_type.lower() == 'netf':
            occlusion = torch.exp(-density * c * deltaT) # Ng x Nr x (Ns*Ns)
            # transmittance = torch.cat([torch.ones([1, occlusion.shape[2]]), occlusion + 1e-7])
            transmittance = torch.cumprod(
                torch.cat([torch.ones([occlusion.shape[0], 1, occlusion.shape[2]]), occlusion+1e-7], 1), 1
            )[:, :-1, :] # cummulative production. Ng x Nr x (Ns * Ns)
            density = density.view(-1, num_r*self.args.num_sampling_points**2) # Ng x Na
            transmittance = transmittance.view(-1, num_r*self.args.num_sampling_points**2)
            rho_density = torch.sum(density * transmittance * rho, dim=0) * c * deltaT

        elif self.args.rendering_type.lower() == 'nlos-neus':
            alpha = 1 - torch.exp(-density * c * deltaT) # Ng x Nr x (Ns*Ns)
            transmittance = torch.cumprod(
                torch.cat([torch.ones([alpha.shape[0], 1, alpha.shape[2]]), 1-alpha+1e-7], 1), 1
            )[:, :-1, :] # cummulative production. Ng x Nr x (Ns * Ns)
            alpha = alpha.view(-1, num_r*self.args.num_sampling_points**2)
            transmittance = transmittance.view(-1, num_r*self.args.num_sampling_points**2)

            alpha = 1 - torch.exp(-density * c * deltaT)
            # occlusion = torch.exp(-density * c * deltaT)
            transmittance = torch.cumprod(torch.cat([torch.ones([1, alpha.shape[1]]), 1 - alpha + 1e-7], 0), 0)[:-1, :]
            alpha = alpha.view(-1) # Na
            transmittance = transmittance.view(-1) # Na
            rho_density = torch.sum(alpha * transmittance * rho, dim=0)

        if out_separately:
            return rho_density, density.view(-1), rho # Na,
        else:
            return rho_density # Na,

    def estimate_rho_w_no_occlusion(self, input_points_ori, current_camera_grid_positions, c, deltaT, scaling_modifier=1.0, out_separately=False):
        gaussian_pdf = self.estimate_gaussian_pdf(input_points_ori, scaling_modifier) # Ng x Na
        opacity = self.get_opacity

        shs_view = self.get_features.transpose(1, 2).view(-1, 1, (self.max_sh_degree+1)**2)
        dir_pp = self.get_mu - current_camera_grid_positions.unsqueeze(0) # Ng, 3
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)

        sh2rho = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
        rho = torch.clamp_min(sh2rho + 0.5, 0.0) # Ng by 1

        density = torch.sum(gaussian_pdf * opacity, dim=0)
        # print(rho.shape)
        rho_density = torch.sum(gaussian_pdf * opacity * rho, dim=0) # Na

        if out_separately:
            return rho_density, density.view(-1), rho # Na,
        else:
            return rho_density # Na,

    def batch_estimate_rho_w_no_occlusion(self, input_points_ori, camera_grid_positions, c, deltaT, scaling_modifier=1.0, out_separately=False):
        # camera_grid_positions : 3, Nx*Ny
        camera_grid_positions = camera_grid_positions.permute(1, 0) # Nx*Ny, 3
        gaussian_pdf = self.estimate_gaussian_pdf(input_points_ori, scaling_modifier) # Ng x Na
        opacity = self.get_opacity

        shs_view = self.get_features.transpose(1, 2).view(-1, 1, (self.max_sh_degree+1)**2)
        dir_pp = self.get_mu.unsqueeze(1) - camera_grid_positions.unsqueeze(0) # Ng, Nx*Ny, 3
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=2, keepdim=True) # Ng, Nx*Ny, 3

        sh2rho = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized) # Ng, Nx*Ny, 1
        rho = torch.clamp_min(sh2rho + 0.5, 0.0) # Ng, Nx*Ny, 1

        # gaussian_pdf : Ng x Na
        # opacity: Ng x 1
        # rho: Ng x Nx*Ny x 1

        density = torch.sum(gaussian_pdf * opacity, dim=0)
        rho_density = torch.sum(gaussian_pdf.unsqueeze(1) * opacity.unsqueeze(1) * rho.unsqueeze(2), dim=0) # (Nx*Ny), Na, 1

        if out_separately:
            return rho_density, density.view(-1), rho
        else:
            return rho_density # (Nx * Ny), Na, 1

    ### Density control based on MCMC-gs
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    def densification_postfix(self, new_mu, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, reset_params=True):
        d = {"mu": new_mu,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._mu = optimizable_tensors["mu"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]


    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {"mu": self._mu,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling" : self._scaling,
            "rotation" : self._rotation}

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if inds is not None:
                #### Make original µs' momentum as 0.
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        self._mu = optimizable_tensors["mu"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        torch.cuda.empty_cache()

        return optimizable_tensors

    def _update_params(self, idxs, ratio):
        new_opacity, new_scaling = compute_relocation_cuda(
            opacity_old=self.get_opacity[idxs, 0],
            scale_old=self.get_scaling[idxs],
            N=ratio[idxs, 0] + 1
        )
        new_opacity = torch.clamp(new_opacity.unsqeeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling.reshape(-1, 3))

        return self._mu[idxs], self._features_dc[idxs], self._features_rest[idxs], new_opacity, new_scaling, self._rotation[idxs]

    def _sample_alives(self, probs, num, alive_indices):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)

        sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1) # 각 sample idxs의 빈도.
        return sampled_idxs, ratio


    # Relocate Dead Gaussians to some live Gaussians!
    def relocate_gs(self, dead_mask=None):
        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        probs = (self.get_opacity[alive_indices, 0])
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        ######## Relocation
        ######## We should compensate the accumulated Gaussians' parameters.
        (
            self._mu[dead_indices],
            self._features_dc[dead_indices],
            self._features_rest[dead_indices],
            self._opacity[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices]
        ) = self._update_params(reinit_idx, ratio=ratio)

        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]

        self.replace_tensors_to_optimizer(inds=reinit_idx)

    def add_new_gs(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        probs = self.get_opacity.squeeze(-1)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (
            new_mu,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation
        ) = self._update_params(add_idx, ratio=ratio)

        self._opacity[add_idx] = new_opacity
        self._scaling[add_idx] = new_scaling

        self.densification_postfix(new_mu, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, reset_params=False)
        self.replace_tensors_to_optimizer(inds=add_idx)

        return num_gs

