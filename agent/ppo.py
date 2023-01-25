import os
import warnings
import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from utils.utils import set_seed
from utils import clip_grad_norms
from nets.actor_network import Actor
from nets.critic_network import Critic
from utils import torch_load_cpu, get_inner_model
from utils.logger import log_to_tb_train, log_to_val
import copy
from problems.gpso_np import GPSO_numpy
from problems.dmspso import DMS_PSO_np
from utils.make_dataset import make_dataset
from rollout import rollout


# memory for recording transition during training process
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]


def lr_sd(epoch, opts):
    return opts.lr_decay ** epoch


class PPO:
    def __init__(self, opts,vector_env):

        # figure out the options
        self.opts = opts
        # the parallel environment
        self.vector_env=vector_env
        # figure out the actor network
        self.actor = Actor(
            embedding_dim = opts.embedding_dim,
            hidden_dim = opts.hidden_dim,
            n_heads_actor = opts.encoder_head_num,
            n_heads_decoder = opts.decoder_head_num,
            n_layers = opts.n_encode_layers,
            normalization = opts.normalization,
            v_range = opts.v_range,
            node_dim=opts.node_dim,
            hidden_dim1=opts.hidden_dim1_actor,
            hidden_dim2=opts.hidden_dim2_actor,
            no_attn=opts.no_attn,
            no_eef=opts.no_eef,
            max_sigma=opts.max_sigma,
            min_sigma=opts.min_sigma,
        )
        
        if not opts.test:
            # for the sake of ablation study, figure out the input_dim for critic according to setting
            if opts.no_attn and opts.no_eef:
                input_critic=opts.node_dim
            elif opts.no_attn and not opts.no_eef:
                input_critic=3*opts.node_dim
            elif opts.no_eef and not opts.no_attn:
                input_critic=opts.node_dim
            else:
                # GLEET(default) setting, share the attention machanism between actor and critic
                input_critic=opts.embedding_dim
            # figure out the critic network
            self.critic = Critic(
                input_dim = input_critic,
                hidden_dim1 = opts.hidden_dim1_critic,
                hidden_dim2 = opts.hidden_dim2_critic,
            )

            # figure out the optimizer
            self.optimizer = torch.optim.Adam(
                [{'params': self.actor.parameters(), 'lr': opts.lr_model}] +
                [{'params': self.critic.parameters(), 'lr': opts.lr_model}])
            # figure out the lr schedule
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, opts.lr_decay, last_epoch=-1,)

        if opts.use_cuda:
            # move to cuda
            self.actor.to(opts.device)
            if not opts.test:
                self.critic.to(opts.device)


    # load model from load_path
    def load(self, load_path):

        assert load_path is not None
        load_data = torch_load_cpu(load_path)

        # load data for actor
        model_actor = get_inner_model(self.actor)
        model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('actor', {})})

        if not self.opts.test:
            # load data for critic
            model_critic = get_inner_model(self.critic)
            model_critic.load_state_dict({**model_critic.state_dict(), **load_data.get('critic', {})})
            # load data for optimizer
            self.optimizer.load_state_dict(load_data['optimizer'])
            # load data for torch and cuda
            torch.set_rng_state(load_data['rng_state'])
            if self.opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # done
        print(' [*] Loading data from {}'.format(load_path))

    # save trained model
    def save(self, epoch):
        print('Saving model and state...')
        torch.save(
            {
                'actor': get_inner_model(self.actor).state_dict(),
                'critic': get_inner_model(self.critic).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    # change working mode to evaling
    def eval(self):
        torch.set_grad_enabled(False)  ##
        self.actor.eval()
        if not self.opts.test: self.critic.eval()

    # change working mode to training
    def train(self):
        torch.set_grad_enabled(True)  ##
        self.actor.train()
        if not self.opts.test: self.critic.train()


    def start_training(self, tb_logger):
        train(0, self, tb_logger)

# inference for training
def train(rank, agent, tb_logger):  
    print("begin training")
    opts = agent.opts
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    # move optimizer's data onto chosen device
    for state in agent.optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(opts.device)


    # generatate the train_dataset and test_dataset
    set_seed(opts.train_dataset_seed)
    training_dataloader = make_dataset(dim=opts.dim,
                                       batch_size=opts.batch_size,
                                       max_x=opts.max_x,
                                       min_x=-opts.max_x,
                                       num_samples=opts.epoch_size,
                                       problem_id=opts.problem,
                                       shifted=opts.shift,
                                       rotated=opts.rotate
                                       )
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


    best_epoch=None
    best_avg_best_cost=None
    best_epoch_list=[]
    mean_per_list=[]
    sigma_per_list=[]
    pre_step=0

    # modify the learning ray after resume(if needed)
    if opts.resume:
        for e in range(opts.epoch_start):
            agent.lr_scheduler.step(e)

    stop_training=False
    # Start the actual training loop
    for epoch in range(opts.epoch_start, opts.epoch_end):
        # Training mode
        set_seed()
        agent.train()
        # agent.lr_scheduler_critic.step(epoch)
        # agent.lr_scheduler_actor.step(epoch)
        agent.lr_scheduler.step(epoch)

        # logging
        if rank == 0:
            print('\n\n')
            print("|",format(f" Training epoch {epoch} ","*^60"),"|")
            print("Training with actor lr={:.3e} critic lr={:.3e} for run {}".format(agent.optimizer.param_groups[0]['lr'],
                                                                                     agent.optimizer.param_groups[1]['lr'], opts.run_name) , flush=True)

        # start training
        step = epoch * (opts.epoch_size // opts.batch_size)
        pbar = tqdm(total = (opts.K_epochs) * (opts.epoch_size // opts.batch_size) * ((opts.max_fes // opts.population_size) // opts.n_step) ,
                    disable = opts.no_progress_bar or rank!=0, desc = 'training',
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

        for batch_id, batch in enumerate(training_dataloader):
            backbone={
                'PSO':GPSO_numpy,
                'DMSPSO':DMS_PSO_np,
            }.get(opts.backbone,None)
            assert backbone is not None,'Backbone algorithm is currently not supported'
            env_list=[lambda e=p: backbone(dim = opts.dim,
                                             max_velocity = opts.max_velocity,
                                             reward_scale = opts.reward_scale,
                                             ps=opts.population_size,problem=e,
                                             max_fes=opts.max_fes,max_x=opts.max_x,boarder_method=opts.boarder_method,
                                             reward_func=opts.reward_func,w_decay=opts.w_decay) for p in batch]
            envs=agent.vector_env(env_list)
            # train procedule for a batch

            batch_step=train_batch(rank,
                                   envs,
                                   agent,
                                   epoch,
                                   pre_step,
                                   batch,
                                   tb_logger,
                                   opts,
                                   pbar,
                                   batch_id)
            envs.close()
            pre_step += batch_step
            # see if the learning step reach the max_learning_step, if so, stop training
            if pre_step>=opts.max_learning_step:
                stop_training=True
                break
        pbar.close()

        # save new model after one epoch
        if rank == 0 and not opts.distributed:
            if not opts.no_saving and (( opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or \
                                       epoch == opts.epoch_end - 1): agent.save(epoch)
                                       
        if (epoch-opts.epoch_start) % opts.update_best_model_epochs==0 or epoch == opts.epoch_end-1:
            # validate the new model
            avg_best,sigma=rollout(test_dataloader,opts,agent,tb_logger,epoch)
            mean_per_list.append(avg_best)
            sigma_per_list.append(sigma)
            if epoch==opts.epoch_start:
                best_avg_best_cost=avg_best
                best_epoch=epoch
            elif avg_best<best_avg_best_cost:
                best_avg_best_cost=avg_best
                best_epoch=epoch
            best_epoch_list.append(best_epoch)

        # logging
        print('current_epoch:{}, best_epoch:{}'.format(epoch,best_epoch))
        print('best_epoch_list:{}'.format(best_epoch_list))
        print(f'mean_performance:{mean_per_list}')
        print(f'sigma_performance:{sigma_per_list}')
        print(f'best_rl_pso_mean:{mean_per_list[(best_epoch-opts.epoch_start)  // opts.update_best_model_epochs]}')
        print(f'best_rl_pso_std:{sigma_per_list[(best_epoch-opts.epoch_start)  // opts.update_best_model_epochs]}')
        
        if stop_training:
            print('Have reached the maximum learning steps')
            break
    print(best_epoch_list)


def train_batch(
        rank,
        problem,
        agent,
        epoch,
        pre_step,
        batch,
        tb_logger,
        opts,
        pbar,
        batch_id):
    
    # setup
    agent.train()
    memory = Memory()

    population_size = opts.population_size

    # initial instances and solutions
    state=problem.reset()
    state=torch.FloatTensor(state).to(opts.device)
    

    # params for training
    gamma = opts.gamma
    n_step = opts.n_step
    
    K_epochs = opts.K_epochs
    eps_clip = opts.eps_clip
    
    t = 0
    # initial_cost = obj
    done=False
    
    # sample trajectory
    while not done:
        t_s = t
        total_cost = 0
        entropy = []
        bl_val_detached = []
        bl_val = []

        # accumulate transition
        while t - t_s < n_step :  
            
            memory.states.append(state.clone())
            action, log_lh,_to_critic,  entro_p  = agent.actor(state,
                                                    require_entropy = True,
                                                    to_critic=True
                                                    )
            

            memory.actions.append(action.clone())
            memory.logprobs.append(log_lh)
            action=action.cpu().numpy()

            entropy.append(entro_p.detach().cpu())

            baseline_val_detached, baseline_val = agent.critic(_to_critic)
            bl_val_detached.append(baseline_val_detached)
            bl_val.append(baseline_val)


            # state transient
            next_state,rewards,is_end,info = problem.step(action)
            memory.rewards.append(torch.FloatTensor(rewards).to(opts.device))
            # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))

            # store info
            # total_cost = total_cost + gbest_val

            # next
            t = t + 1
            state=torch.FloatTensor(next_state).to(opts.device)
            if is_end.all():
                done=True
                break


        
        # store info
        t_time = t - t_s
        total_cost = total_cost / t_time

        # begin update

        old_actions = torch.stack(memory.actions)
        old_states = torch.stack(memory.states).detach() #.view(t_time, bs, ps, dim_f)
        # old_actions = all_actions.view(t_time, bs, ps, -1)
        # print('old_actions.shape:{}'.format(old_actions.shape))
        old_logprobs = torch.stack(memory.logprobs).detach().view(-1)

        # Optimize PPO policy for K mini-epochs:
        old_value = None
        for _k in range(K_epochs):
            if _k == 0:
                logprobs = memory.logprobs

            else:
                # Evaluating old actions and values :
                logprobs = []
                entropy = []
                bl_val_detached = []
                bl_val = []

                for tt in range(t_time):

                    # get new action_prob
                    _, log_p,_to_critic,  entro_p = agent.actor(old_states[tt],
                                                     fixed_action = old_actions[tt],
                                                     require_entropy = True,# take same action
                                                     to_critic = True
                                                     )

                    logprobs.append(log_p)
                    entropy.append(entro_p.detach().cpu())

                    baseline_val_detached, baseline_val = agent.critic(_to_critic)

                    bl_val_detached.append(baseline_val_detached)
                    bl_val.append(baseline_val)

            logprobs = torch.stack(logprobs).view(-1)
            entropy = torch.stack(entropy).view(-1)
            bl_val_detached = torch.stack(bl_val_detached).view(-1)
            bl_val = torch.stack(bl_val).view(-1)


            # get traget value for critic
            Reward = []
            reward_reversed = memory.rewards[::-1]
            # get next value
            R = agent.critic(agent.actor(state,only_critic = True))[0]

            # R = agent.critic(x_in)[0]
            critic_output=R.clone()
            for r in range(len(reward_reversed)):
                R = R * gamma + reward_reversed[r]
                Reward.append(R)
            # clip the target:
            Reward = torch.stack(Reward[::-1], 0)
            Reward = Reward.view(-1)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = Reward - bl_val_detached

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
            reinforce_loss = -torch.min(surr1, surr2).mean()

            # define baseline loss
            if old_value is None:
                baseline_loss = ((bl_val - Reward) ** 2).mean()
                old_value = bl_val.detach()
            else:
                vpredclipped = old_value + torch.clamp(bl_val - old_value, - eps_clip, eps_clip)
                v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
                baseline_loss = v_max.mean()

            # check K-L divergence (for logging only)
            approx_kl_divergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
            approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0
            # calculate loss
            loss = baseline_loss + reinforce_loss

            # update gradient step
            agent.optimizer.zero_grad()
            loss.backward()

            # Clip gradient norm and get (clipped) gradient norms for logging
            current_step = int(pre_step + t//n_step * K_epochs  + _k)
            grad_norms = clip_grad_norms(agent.optimizer.param_groups, opts.max_grad_norm)

            # perform gradient descent
            agent.optimizer.step()

            # Logging to tensorboard
            if(not opts.no_tb) and rank == 0:
                if current_step % int(opts.log_step) == 0:
                    log_to_tb_train(tb_logger, agent, Reward,R,critic_output, ratios, bl_val_detached, total_cost, grad_norms, memory.rewards, entropy, approx_kl_divergence,
                                    reinforce_loss, baseline_loss, logprobs, opts.show_figs, current_step)

            if rank == 0: pbar.update(1)
            # end update
        

        memory.clear_memory()

    # return learning steps
    return ( t // n_step + 1) * K_epochs

