import os
import time
import argparse
import torch


def get_options(args=None):

    parser = argparse.ArgumentParser(description="GLEET")
    
    # core setting
    parser.add_argument('--train',default=None,action='store_true',help='switch to train mode')
    parser.add_argument('--test',default=None,action='store_true', help='switch to inference mode')
    parser.add_argument('--run_name',default='test',help='name to identify the run')
    parser.add_argument('--load_path', default = None, help='path to load model parameters and optimizer state from')
    parser.add_argument('--resume', default = None, help='resume from previous checkpoint file')
    parser.add_argument('--problem', default=1,type=int,
                        choices = [1,2,3,4,5,6,7,8,9,10])
    parser.add_argument('--backbone',default='PSO',choices=['PSO','DMSPSO'],help='the backbone algorithm for the train')
    
    parser.add_argument('--is_linux',default=False,help='for the usage of parallel environment, os should be known by program')     
    # for ablation study
    parser.add_argument('--no_attn',action='store_true',default=False,help='whether the network has attention mechanism or not')
    parser.add_argument('--no_eef',action='store_true',default=False,help='whether the state has exploitation&exploration features ')

    

    # environment settings
    parser.add_argument('--reward_scale', type = float, default= 100,help='reward=origin reward * reward_scale, make the reward bigger for easier training') 
    parser.add_argument('--population_size', type = int, default= 100,help='population size use in backbone algorithm')  # recommend 100
    parser.add_argument('--max_velocity', type = float, default=10,help='the upper bound of velocity in PSO family algorithm')
    parser.add_argument('--dim', type=int, default=10,help='dimension of the sovling problems')
    parser.add_argument('--max_x',default=100,help='the upper bound of the searing range')
    parser.add_argument('--boarder_method',default='periodic',choices=['clipping','random','periodic','reflect'])
    # parser.add_argument('--reward_stop',default=False)    # deprecated
    parser.add_argument('--reward_func',default='direct',choices=['direct','relative','triangle','11'],help='several reward functions for comparison')
    parser.add_argument('--w_decay',default=True,help='whether to use w_decay in backbone algorithm, which is recommended')
    parser.add_argument('--shift',default=True,help='whether to generation the dataset with shifted problems')
    parser.add_argument('--rotate',default=True,help='whether to generation the dataset with rotated problems')

    # parameters in framework
    parser.add_argument('--no_cuda', action='store_true', help='disable GPUs')
    parser.add_argument('--no_tb', action='store_true', help='disable Tensorboard logging')
    parser.add_argument('--show_figs', action='store_true', help='enable figure logging')
    parser.add_argument('--no_saving', action='store_true', help='disable saving checkpoints')
    parser.add_argument('--use_assert', action='store_true', help='enable assertion')
    parser.add_argument('--no_DDP', action='store_true', help='disable distributed parallel')
    parser.add_argument('--seed', type=int, default=1024, help='random seed to use')


    # add test_set_seed , train_set_seed
    parser.add_argument('--test_dataset_seed',default=999,help='the random seed for generating test dataset')
    parser.add_argument('--train_dataset_seed',default=42,help='the random seed for generating train dataset')


    # Net(Attention Aggragation) parameters
    parser.add_argument('--v_range', type=float, default=6., help='to control the entropy')
    parser.add_argument('--encoder_head_num', type=int, default=4, help='head number of encoder')
    parser.add_argument('--decoder_head_num', type=int, default=4, help='head number of decoder')
    parser.add_argument('--critic_head_num', type=int, default=4, help='head number of critic encoder')
    parser.add_argument('--embedding_dim', type=int, default=16, help='dimension of input embeddings') # 
    parser.add_argument('--hidden_dim', type=int, default=16, help='dimension of hidden layers in Enc/Dec') # 减小
    parser.add_argument('--n_encode_layers', type=int, default=1, help='number of stacked layers in the encoder') # 减小一点
    parser.add_argument('--normalization', default='layer', help="normalization type, 'layer' (default) or 'batch'")
    parser.add_argument('--node_dim',default=9,type=int,help='feature dimension for backbone algorithm')
    parser.add_argument('--hidden_dim1_critic',default=32,help='the first hidden layer dimension for critic')
    parser.add_argument('--hidden_dim2_critic',default=16,help='the second hidden layer dimension for critic')
    parser.add_argument('--hidden_dim1_actor',default=32,help='the first hidden layer dimension for actor')
    parser.add_argument('--hidden_dim2_actor',default=8,help='the first hidden layer dimension for actor')
    parser.add_argument('--output_dim_actor',default=1,help='output action dimension for actor')
    parser.add_argument('--lr_decay', type=float, default=0.9862327, help='learning rate decay per epoch',choices=[0.998614661,0.9862327])
    parser.add_argument('--max_sigma',default=0.7,type=float,help='upper bound for actor output sigma')
    parser.add_argument('--min_sigma',default=0.01,type=float,help='lowwer bound for actor output sigma')

    # Training parameters
    parser.add_argument('--max_learning_step',default=4000000,help='the maximum learning step for training')
    parser.add_argument('--RL_agent', default='ppo', choices = ['ppo'], help='RL Training algorithm')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor for future rewards')
    parser.add_argument('--K_epochs', type=int, default=3, help='mini PPO epoch')
    parser.add_argument('--eps_clip', type=float, default=0.1, help='PPO clip ratio')
    parser.add_argument('--T_train', type=int, default=1800, help='number of itrations for training')
    parser.add_argument('--n_step', type=int, default=10, help='n_step for return estimation')
    parser.add_argument('--batch_size', type=int, default=16,help='number of instances per batch during training')
    parser.add_argument('--epoch_start', type=int, default=0, help='start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--epoch_end', type=int, default=100, help='maximum training epoch')
    parser.add_argument('--epoch_size', type=int, default=1024, help='number of instances per epoch during training')
    parser.add_argument('--lr_model', type=float, default=4e-5, help="learning rate for the actor network")
    parser.add_argument('--max_grad_norm', type=float, default=0.1, help='maximum L2 norm for gradient clipping')
    
    # validate parameters
    parser.add_argument('--update_best_model_epochs',type=int,default=3,help='update the best model every n epoch')
    parser.add_argument('--val_size', type=int, default=128, help='number of instances for validation/inference')
    parser.add_argument('--per_eval_time',type=int,default=10,help='per problem eval n time')

    # logs/output settings
    parser.add_argument('--no_progress_bar', action='store_true', help='disable progress bar')
    parser.add_argument('--log_dir', default='logs', help='directory to write TensorBoard information to')
    parser.add_argument('--log_step', type=int, default=50, help='log info every log_step gradient steps')
    parser.add_argument('--output_dir', default='outputs', help='directory to write output models to')
    parser.add_argument('--checkpoint_epochs', type=int, default=1, help='save checkpoint every n epochs (default 1), 0 to save no checkpoints')

    opts = parser.parse_args(args)
    
    if opts.backbone=='PSO':
        opts.population_size=100
    elif opts.backbone=='DMSPSO':
        opts.population_size=99
    # opts.max_velocity=0.1*opts.max_x
    
    if not opts.no_attn:
        opts.lr_model=4e-5
    else:
        opts.lr_model=1e-4
    
    # figure out whether to use distributed training if needed (deprecated)
    # opts.world_size = torch.cuda.device_count()
    # opts.distributed = (torch.cuda.device_count() > 1) and (not opts.no_DDP)
    opts.world_size = 1
    opts.distributed = False
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '4869'
    # processing settings
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name,time.strftime("%Y%m%dT%H%M%S")) \
        if not opts.resume else opts.resume.split('/')[-2]
    opts.save_dir = os.path.join(
        opts.output_dir,
        opts.backbone,
        "function{}_{}".format(opts.problem, opts.dim),
        opts.run_name
    ) if not opts.no_saving else None

    return opts
