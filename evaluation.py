from train import *

def construct_eval_coords(args, data_kwargs):
    volume_size = data_kwargs['volume_size']
    unit_distance = volume_size / (args.eval_resolution - 1)
    pmin, pmax = data_kwargs['pmin'].cpu().numpy(), data_kwargs['pmax'].cpu().numpy()
    xv = np.linspace(pmin[0], pmax[0], args.eval_resolution)
    yv = np.linspace(pmin[1], pmax[1], args.eval_resolution)
    zv = np.linspace(pmin[2], pmax[2], args.eval_resolution)

    coords = np.stack(np.meshgrid(xv, yv, zv, indexing='ij'),-1)
    coords = coords.reshape([-1,3])
    coords = torch.from_numpy(coords.astype(np.float32)).to(data_kwargs['camera_grid_positions'].device)
    return coords

def make_middle_cam_pos(camera_grid_positions):
    _, Ns = camera_grid_positions.shape
    N = int(math.sqrt(Ns))
    middle = N//2
    m_cam_pos = camera_grid_positions.view(N, N, 3)[middle, middle]
    return m_cam_pos

def make_eval_kwargs(args, data_kwargs):
    eval_coords = construct_eval_coords(args, data_kwargs)
    m_cam_pos = make_middle_cam_pos(data_kwargs['camera_grid_positions'])
    eval_kwargs = {
        'eval_coords': eval_coords,
        'eval_cam_pos': m_cam_pos,
    }
    return eval_kwargs


def evaluation(args, optim_args, load_path, device, current_iter=25000):
    data_kwargs, nlos_data, camera_grid_positions, index = make_data_kwargs(args, device)

    ## center cam pos
    m_cam_pos = make_middle_cam_pos(camera_grid_positions)

    model = create_model(args, data_kwargs, optim_args, device, evaluation=True)
    model.restore(load_path, optim_args)
    # gaussian2volume(model, m_cam_pos, process_batch=args.eval_proc_batch, resolution=args.eval_resolution, mode='voxel')
    # gaussian2volume(args, model, data_kwargs, m_cam_pos, resolution=args.eval_resolution, mode='mesh')
    coords = construct_eval_coords(args, data_kwargs)

    eval_kwargs = make_eval_kwargs(args, data_kwargs)
    coords = eval_kwargs['eval_coords']
    m_cam_pos = eval_kwargs['eval_cam_pos']
    gaussian2volume(args, model, coords, data_kwargs, m_cam_pos, current_iter)

if __name__ == "__main__":
    device='cuda:0'
    optim_args = OptimizationParams()
    args = Config()
    model_save_rel_dir = args.model_save_rel_dir
    model_dir = model_save_rel_dir
    load_path = os.path.join(model_dir, 'current_iter25000.pt')
    evaluation(args, optim_args, load_path, device)