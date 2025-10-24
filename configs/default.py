import argparse

class Config:
    def __init__(self):

        self.train = True # if this is False, only conduct evaluation

        self.rng = 0
        self.datadir = './data/zaragozadataset/zaragoza256_preprocessed.mat'
        self.dataset_type = 'zaragoza256'
        self.scene = 'zaragoza_bunny'
        self.gt_times = 100
        self.save_fig = True
        self.cuda = 0
        self.occlusion = False
        self.epoches = 1000
        self.start = 100
        self.end = 300
        self.num_sampling_points = 32
        self.expname = 'zaragoza-bunny-256'
        self.basedir = './logs'


        self.config = 'config'
        self.model_save_rel_dir = 'model'
        self.save_model_interval = 5000
        self.save_hist_fig_interval = 500
        self.print_interval = 100

        #### Gaussian Instance Init
        self.sh_degree = 3
        self.init_gaussian_num = 2000
        self.init_sample_margin = 0.1
        self.space_carving_init = True
        self.carving_volume_size = 64
        self.space_carving_ratio = 0.99
        self.scaling_modifier = 1.

        self.rendering_type = 'netf'
        
        ## CUDA rendering
        self.use_cuda_renderer = False  # Set to True to use CUDA-accelerated ray-based rendering

        ## evaluation
        self.eval_resolution = 256

        # if CPU_DEBUG:
        #     self.start = 150
        #     self.end = 250
        #     self.num_sampling_points = 4
        #     self.carving_volume_size = 4
        #     self.save_model_interval = 10
        #     self.init_gaussian_num = 500
        #     self.eval_resolution = 256

    def to_namespace(self):
        return argparse.Namespace(**self.__dict__)

class OptimizationParams:
    def __init__(self):
        self.iterations = 50_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 50_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2

        ##### Densitification params
        self.mcmc_densification_flag = False
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 25_000
        self.densify_grad_threshold = 0.0002
        self.cap_max = 100000


        ##### Loss coef
        self.regularization = False
        self.scale_reg = 0.01
        self.opacity_reg = 0.01
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        self.warmup_iter = 500

        ##### Indexing transient images
        self.nlos_data_random_indexing = True


        # if CPU_DEBUG:
        #     self.mcmc_densification_flag = False

        #     self.iterations = 3_000
        #     self.position_lr_max_steps = 3_000
        #     self.densify_until_iter = 1_500

        #     self.cap_max = 10_000
        #     self.warmup_iter = 100