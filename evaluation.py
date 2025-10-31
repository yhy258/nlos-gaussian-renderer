from train import *


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

if __name__ == "__main__":
    device='cuda:0'
    optim_args = OptimizationParams()
    args = Config()
    model_save_rel_dir = args.model_save_rel_dir
    model_dir = model_save_rel_dir
    load_path = os.path.join(model_dir, 'current_iter25000.pt')
    evaluation(args, optim_args, load_path, device)