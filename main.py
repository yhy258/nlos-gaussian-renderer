import torch
from nlos_helpers import *
from configs.default import Config, OptimizationParams
from train import *
from evaluation import *

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
    load_path = os.path.join(model_dir, 'current_iter5000.pt')
    evaluation(args, optim_args, load_path, device)