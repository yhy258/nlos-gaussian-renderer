import os, sys
import math
import numpy as np
import json
import random
import itertools
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt # not used in this code.
from nlos_helpers import *
from configs.default import Config, OptimizationParams
from gaussian_model.gaussian_model import GaussianModel
from gaussian_model.gaussian_utils import init_rand_points, sample_from_feasible_space_jittering
from data.data_loader import load_zaragoza256_data


def random_seed(args):
    seed = args.rng
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cycle_random_pairs(M, N):
    """
    Generator
    """
    all_pairs = list(itertools.product(range(M), range(N)))

    while True:
        random.shuffle(all_pairs)
        for m, n in all_pairs:
            yield m, n

@torch.no_grad()
def data_shuffle(nlos_data, camera_grid_positions, device):
    L, M, N = nlos_data.shape

    nlos_data = nlos_data.reshape(L, -1)
    camera_grid_positions = torch.from_numpy(camera_grid_positions).float().to(device)
    index = torch.linspace(0, M * N - 1, M * N, dtype=torch.float, device=device).reshape(1, -1)
    full_data = torch.cat((nlos_data, camera_grid_positions, index), axis = 0)
    full_data = full_data[:,torch.randperm(full_data.size(1))]
    nlos_data = full_data[0:L,:].view(L,M,N)
    camera_grid_positions = full_data[L:-1,:]
    index = full_data[-1,:]
    del full_data
    """
    Output:
        nlos_data (torch.Tensor) (L, M, N)
        camera_grid_positions (torch.Tensor) (MN, 3)
        index (torch.Tensor) (MN,)
    """
    return nlos_data, camera_grid_positions, index


def update_lr(optimizer, args):
    # clear varible
    # update learning rate
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 0.0000001:
            param_group['lr'] = param_group['lr'] * args.lr_decay
            learning_rate = param_group['lr']
            print('learning rate is updated to ',learning_rate)
    return 0

def create_model(args, data_kwargs, optim_args, device, evaluation=False):
    # Create Gaussian Model
    model = GaussianModel(args, device)

    # initialize points and rhos
    if evaluation or args.space_carving_init == False:
        # when we evaluate them, we initialize the gaussians using random init, since space-carving requires computations to some degree.
        points, rhos = init_rand_points(args, data_kwargs, rho_scale=0.2 ,margin=args.init_sample_margin, device=device)
    else:
        points, rhos = sample_from_feasible_space_jittering(args, data_kwargs, margin=args.init_sample_margin, device=device)
    # Initialize parameters.
    print(rhos.device)
    model.create_params(points, rhos, data_kwargs['pmin'], data_kwargs['pmax'])

    # Training setup (optimizer, ...)
    model.training_setup(optim_args)

    return model


def make_data_kwargs(args, device):
    print('Data Device: ', device)
    #### zaragoza data load. (I only considered zaragoza nlos data)
    nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c = load_zaragoza256_data(args.datadir)
    ### what is the pmin and pmax? - Max, Min spatial or angular values in the target volume
    pmin = volume_position - volume_size / 2
    pmax = volume_position + volume_size / 2
    pmin = np.concatenate((pmin,np.array([0, -np.pi])), axis = 0)
    pmax = np.concatenate((pmax,np.array([np.pi, 0])), axis = 0)
    ### get volume box points
    box_point = volume_box_point(volume_position, volume_size)


    # Device
    nlos_data = torch.tensor(nlos_data, dtype=torch.float, device=device)
    pmin = torch.tensor(pmin, dtype=torch.float, device=device)
    pmax = torch.tensor(pmax, dtype=torch.float, device=device)
    camera_grid_size = torch.tensor(camera_grid_size, dtype=torch.float, device=device)
    volume_position = torch.tensor(volume_position, dtype=torch.float, device=device)
    box_point = torch.tensor(box_point, dtype=torch.float, device=device)


    ## data shuffler
    nlos_data, camera_grid_positions, index = data_shuffle(nlos_data, camera_grid_positions, device)

    print('nlos_data device: ', nlos_data.device)
    print('volume_position device: ', volume_position.device)
    print('camera_grid_positions device: ', camera_grid_positions.device)

    # make kwargs
    data_kwargs = {
        'nlos_data': nlos_data, # L x M x N
        'index': index, # MN
        'camera_grid_positions': camera_grid_positions, # 3 x MN
        'camera_grid_size': camera_grid_size, # 2,
        'volume_position': volume_position, # 3,
        'volume_size': volume_size, # float
        'volume_box_point': box_point,
        'deltaT': deltaT, # float
        'c': c, # float
        'pmin': pmin,
        'pmax': pmax
    }
    return data_kwargs, nlos_data, camera_grid_positions, index

def make_optim_kwargs(args):
    criterion = torch.nn.MSELoss(reduction='mean')

    N_iters = args.epoches
    optim_kwargs = {'criterion': criterion, 'N_iters': N_iters}
    return optim_kwargs

def warmup_learn_func(args, optim_args, model, data_kwargs, optim_kwargs, device):
    M, N = optim_kwargs['M'], optim_kwargs['N']

    pair_generator = cycle_random_pairs(M, N)
    while optim_kwargs['current_iter'] <= optim_args.warmup_iter:
        m, n = next(pair_generator)
        model.update_learning_rate(optim_kwargs['current_iter'])
        model.optimizer.zero_grad()
        optim_kwargs['m'], optim_kwargs['n'] = m, n
        loss, equal_loss = compute_loss(args, model, data_kwargs, optim_kwargs, device)
        if optim_args.regularization:
            # reg1 = optim_args.opacity_reg * torch.abs(model.get_opacity).mean()
            # reg2 = optim_args.scale_reg * torch.abs(model.get_scaling).mean()
            loss = loss + optim_args.opacity_reg * torch.abs(model.get_opacity).mean()
            loss = loss + optim_args.scale_reg * torch.abs(model.get_scaling).mean()

        # print(f"Trainsient point ({m}, {n}) Loss: ", loss)
        loss.backward()
        with torch.no_grad():
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none = True)
            ######### If we want.. we can conduct Brownian motion!
            ######### -> SGLD.

        optim_kwargs['current_iter'] += 1

    dt = time.time()-optim_kwargs['prev_time']
    # warmup completed time
    print(f"Complete Warmup Iterations")
    print(f"Time: {dt},  The Warmup Final Loss: {loss.item()}")
    optim_kwargs['prev_time'] = time.time()
    return model, optim_kwargs

def learn_func(args, optim_args, model, data_kwargs, optim_kwargs, device, gpu_verbose=False):
    """
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
        optim_kwargs:
            prev_time: The start time of the training
            M, m, N, n: indices for the current step
            where N is the number of columns, m is the current index of rows, and n is the current index of columns.
            (j: target histogram. According to the given setting, we can accumulate the loss using the indices j
            N_iters: epochs
            criterion: the main loss function of the model
            total_iter: The total number of iterations
            current_iter: The current iteration
    """

    def learn_one_iter(m, n):
        vram_log = ""
        model.update_learning_rate(optim_kwargs['current_iter'])
        model.optimizer.zero_grad()
        optim_kwargs['m'], optim_kwargs['n'] = m, n
        loss, equal_loss = compute_loss(args, model, data_kwargs, optim_kwargs, device)
        if optim_args.regularization:
            # reg1 = optim_args.opacity_reg * torch.abs(model.get_opacity).mean()
            # reg2 = optim_args.scale_reg * torch.abs(model.get_scaling).mean()
            loss = loss + optim_args.opacity_reg * torch.abs(model.get_opacity).mean()
            loss = loss + optim_args.scale_reg * torch.abs(model.get_scaling).mean()
        # print(f"Trainsient point ({m}, {n}) Loss: ", loss)
        loss.backward()

        with torch.no_grad():
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none = True)
            ######### (FUTURE WORK)
            ######### If we want.. we can conduct Brownian motion!
            ######### -> SGLD.

            # if (n % 16 == 0):
            if optim_kwargs['current_iter'] % args.print_interval == 0:
                dt = time.time()-optim_kwargs['prev_time']
                if gpu_verbose:
                    # --- VRAM Monitoring ---
                    allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
                    reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
                    vram_log = f"  VRAM(Alloc/Reserved): {allocated_gb:.2f}/{reserved_gb:.2f} GB"

                print(optim_kwargs['current_iter'], '/', optim_kwargs['total_iter'], 'iter  ', m, '/', data_kwargs['nlos_data'].shape[1],'  ', n,'/',data_kwargs['nlos_data'].shape[2], '  histgram loss: ',loss.item(), 'time: ', dt, vram_log)

                # print(i,'/',optim_kwargs['N_iters'],'iter  ', m,'/', data_kwargs['nlos_data'].shape[1],'  ', n,'/',data_kwargs['nlos_data'].shape[2], '  histgram loss: ',loss.item(), 'time: ', dt, vram_log)
                optim_kwargs['prev_time'] = time.time()
                if optim_kwargs['current_iter'] == 48:
                    total_time = dt * optim_kwargs['total_iter'] / 16 / 60 / 60
                    print('total time: ', total_time, ' hours')

            if optim_kwargs['current_iter'] % args.save_model_interval == 0:
                save_model(args, model, optim_kwargs['current_iter'])

            optim_kwargs['current_iter'] += 1
            if optim_kwargs['current_iter'] % 1000:
                model.oneupSHdegree()

            if optim_args.mcmc_densification_flag:
                if optim_kwargs['current_iter'] < optim_args.densify_until_iter and optim_kwargs['current_iter'] > optim_args.densify_from_iter and optim_kwargs['current_iter'] % optim_args.densification_interval == 0:
                    dead_mask = (model.get_opacity <= 0.005).squeeze(-1)
                    model.relocate_gs(dead_mask=dead_mask)
                    model.add_new_gs(cap_max=optim_args.cap_max)


        if optim_kwargs['current_iter'] > optim_kwargs['total_iter']:
            complete = True
            return complete
        else:
            return False

    M, N = optim_kwargs['M'], optim_kwargs['N']
    if optim_args.nlos_data_random_indexing:
        while True:
            pair_generator = cycle_random_pairs(M, N)
            m, n = next(pair_generator)
            complete = learn_one_iter(m, n)
            if complete:
                return model, optim_kwargs, complete
    else:
        for m in range(0, M):
            for n in range(0, N):
                complete = learn_one_iter(m, n)
                if complete:
                    return model, optim_kwargs, complete



def train(args, optim_args, device):
    args.scaling_modifier = 1.0


    # print args.
    print('----------------------------------------------------')
    print('Loaded: ' + args.datadir)
    print('dataset_type: ' + args.dataset_type)
    print('gt_times: ' + str(args.gt_times))
    print('save_fig: ' + str(args.save_fig))
    print('cuda: ' + str(args.cuda))
    print('start: ' + str(args.start))
    print('end: ' + str(args.end))
    print('num_sampling_points: ' + str(args.num_sampling_points))
    print('carving_volume_size: ' + str(args.carving_volume_size))
    print('----------------------------------------------------')


    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    extrapath = './model/'
    if not os.path.exists(extrapath):
        os.makedirs(extrapath)
    extrapath = './figure/'
    if not os.path.exists(extrapath):
        os.makedirs(extrapath)
    extrapath = './figure/test'
    if not os.path.exists(extrapath):
        os.makedirs(extrapath)

    # set the dictionary inputs
    """
    These [..._kwargs] are the dictionaries that contain values dynamically changing or defined in the run code.
    On the other hand, [..._args] are the arguments that contain the code settings.
        data_kwargs:
            nlos_data: NLOS data.
            index: the indices for the shuffled data
            camera_grid_positions: The position of the camera grid (visible wall?); (Na x 3) : torch.Tensor
            camera_grid_size: The size of the camera grid : scalar
            volume_position: The center position of the hidden volume: (3,): torch.Tensor
            volume_size: The size of the hidden volume: Scalar
            volume_box_point: The vertex of the volume cube.
            deltaT: The discrete time interval in this setting
            c: The speed of the light
            pmin and pmax: the range of the volume coordinate
        optim_kwargs:
            prev_time: The start time of the training
            M, m, N, n: indices for the current step
            where N is the number of columns, m is the current index of rows, and n is the current index of columns.
            (j: target histogram. According to the given setting, we can accumulate the loss using the indices j
            N_iters: epochs
            criterion: the main loss function of the model
            optimizer: the optimizer of the model
            global_step: global step
        eval_kwargs:
            coords: The target volume coordinate mesh
            axes_coords (xv, yv, zv): Each axis coordinate for the mesh
            target_volume_shape (P, Q, R): The number of pixels on each axis
            test_batchsize:
            global_step: global step
    """
    # make data_kwargs: The whole data information.
    ## cf) Why should we get the nlos_data, camera_grid_positions, index separately? (even data_kwargs contains them.)
    ## We will use the parameters to rebalance them (two-stage training).
    ## The actually used data is data_kwargs['nlos_data'], and the global variable 'nlos_data' in the train function would be used to rebalance them.
    print(f'Current Device: {device}')
    data_kwargs, nlos_data, camera_grid_positions, index = make_data_kwargs(args, device)
    print('deltaT: ' + str(data_kwargs['deltaT']))
    # Create model
    model = create_model(args, data_kwargs, optim_args, device)

    L, M, N = nlos_data.shape

    # Make optim_kwargs (criterion, ...)
    optim_kwargs = make_optim_kwargs(args)
    optim_kwargs['M'] = M
    optim_kwargs['N'] = N

    optim_kwargs['total_iter'] = optim_args.iterations
    optim_kwargs['current_iter'] = 1

    # TRAIN
    time0 = time.time()
    optim_kwargs['prev_time'] = time0
    print(' ')
    while True:
        model, optim_kwargs = warmup_learn_func(args, optim_args, model, data_kwargs, optim_kwargs, device)
        model, optim_kwargs, complete = learn_func(args, optim_args, model, data_kwargs, optim_kwargs, device)

        if complete:
            break


def evaluation(args, optim_args, load_path, device):
    data_kwargs, nlos_data, camera_grid_positions, index = make_data_kwargs(args, device)

    ## center cam pos
    cam_pos = data_kwargs['camera_grid_positions']
    _, Ns = cam_pos.shape
    N = int(math.sqrt(Ns))
    middle = N//2
    m_cam_pos = cam_pos.view(N, N, 3)[middle, middle]

    model = create_model(args, data_kwargs, optim_args, device, evaluation=True)
    model.restore(load_path, optim_args)
    # gaussian2volume(model, m_cam_pos, process_batch=args.eval_proc_batch, resolution=args.eval_resolution, mode='voxel')
    gaussian2volume(args, model, data_kwargs, m_cam_pos, resolution=args.eval_resolution, mode='mesh')



if __name__=='__main__':
    optim_args = OptimizationParams()
    args = Config()
    random_seed(args)
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    if args.train:
        train(args, optim_args, device)

    model_save_rel_dir = args.model_save_rel_dir
    model_dir = model_save_rel_dir
    load_path = os.path.join(model_dir, 'current_iter110.pt')
    evaluation(args, optim_args, load_path, device)