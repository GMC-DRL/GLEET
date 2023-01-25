import os
import json
import torch
import pprint
from tensorboardX import SummaryWriter
import warnings
from options import get_options

from agent.ppo import PPO
from utils.utils import set_seed
# DummyVectorEnv for Windows or Linux, SubprocVectorEnv for Linux
from env import DummyVectorEnv,SubprocVectorEnv
import platform


def load_agent(name):
    agent = {
        'ppo': PPO,
    }.get(name, None)
    assert agent is not None, "Currently unsupported agent: {}!".format(name)
    return agent

def run(opts):
    # only one mode can be specified in one time, test or train
    assert opts.train==None or opts.test==None, 'Between train&test, only one mode can be given in one time'
    
    sys=platform.system()
    opts.is_linux=True if sys == 'Linux' else False

    # figure out the max_fes(max function evaluation times), in our experiment, we use 20w for 10D problem and 100w for 30D problem
    if opts.dim==10:
        opts.max_fes=200000
    elif opts.dim==30:
        opts.max_fes=1000000

    # Pretty print the run args
    pprint.pprint(vars(opts))

    # Set the random seed to initialize the network
    set_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tb:
        tb_logger = SummaryWriter(os.path.join(opts.log_dir,opts.RL_agent, "{}_{}".format(opts.problem,
                                                          opts.dim), opts.run_name))

    if not opts.no_saving and not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
        
    # Save arguments so exact configuration can always be found
    if not opts.no_saving:
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

    # Set the device, you can change it according to your actual situation
    opts.device = torch.device("cuda:1" if opts.use_cuda else "cpu")
           
    # Figure out the RL algorithm
    if opts.is_linux:
        agent = load_agent(opts.RL_agent)(opts,SubprocVectorEnv)
    else:
        agent = load_agent(opts.RL_agent)(opts,DummyVectorEnv)

    # Load data from load_path(if provided)
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        agent.load(load_path)

    # Do validation only
    if opts.test:
        # Testing
        from utils.make_dataset import make_dataset
        from rollout import rollout
        # Load the validation datasets
        set_seed(opts.test_dataset_seed)
        test_dataloader=make_dataset(dim=opts.dim,
                                        batch_size=opts.batch_size,
                                        max_x=opts.max_x,
                                        min_x=-opts.max_x,
                                        num_samples=opts.val_size,
                                        problem_id=opts.problem,
                                        shifted=opts.shift,
                                        rotated=opts.rotate
                                        )
        gbest_mean,std=rollout(test_dataloader,opts)
        print(f'func_{opts.problem},gbest_mean:{gbest_mean},std:{std}')
        
    else:
        # Training
        # Resume training if resume_load_path is provided
        if opts.resume:
            epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
            print("Resuming after {}".format(epoch_resume))
            agent.opts.epoch_start = epoch_resume + 1
    
        # Start the actual training loop
        agent.start_training(tb_logger)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # main process
    run(get_options())
