from utils.utils import set_seed
import numpy as np
from problems.dmspso import DMS_PSO_np
from problems.gpso_np import GPSO_numpy
import torch
from tqdm import tqdm
from utils.logger import log_to_val
import os
from env import DummyVectorEnv,SubprocVectorEnv

# interface for rollout
def rollout(dataloader,opts,agent=None,tb_logger=None, epoch_id=0):

    rollout_name=f'func_{opts.problem}_{opts.backbone}'
    if agent:
        rollout_name='GLEET-'+rollout_name
        agent.eval()
    T = opts.max_fes // opts.population_size+1

    # to store the whole rollout process
    cost_rollout=np.zeros((opts.batch_size,T-1))
    
    time_eval=0
    collect_mean=[]
    collect_std=[]
    
    
    # set the same random seed before rollout for the sake of fairness
    set_seed(42)
    for bat_id,batch in enumerate(dataloader):
        # figure out the backbone algorithm
        backbone={
            'PSO':GPSO_numpy,
            'DMSPSO':DMS_PSO_np,
        }.get(opts.backbone,None)
        assert backbone is not None,'Backbone algorithm is currently not supported'
        # see if there is agent to aid the backbone
        origin=True
        if agent:
            origin=False
        env_list=[lambda e=p: backbone(dim = opts.dim,
                                            max_velocity = opts.max_velocity,
                                            reward_scale = opts.reward_scale,
                                            ps=opts.population_size,problem=e,origin=origin,
                                            max_fes=opts.max_fes,max_x=opts.max_x,boarder_method=opts.boarder_method,
                                            reward_func=opts.reward_func,w_decay=opts.w_decay) for p in batch]
        # Parallel environmen SubprocVectorEnv can only be used in Linux
        vector_env=SubprocVectorEnv if opts.is_linux else DummyVectorEnv
        problem=vector_env(env_list)

        # list to store the final optimization result
        collect_gbest=np.zeros((opts.batch_size,opts.per_eval_time))
        for i in range(opts.per_eval_time):
            # reset the backbone algorithm
            is_end=False
            state=problem.reset()
            
            if agent:
                state=torch.FloatTensor(state).to(opts.device)

            time_eval+=1
            
            # visualize the rollout process
            for t in tqdm(range(T), disable = opts.no_progress_bar,
                            desc = rollout_name, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
                
                if agent:
                    # if RL_agent is provided, the action is from the agent.actor
                    action,_,_to_critic = agent.actor(state,to_critic=True)
                    action=action.cpu().numpy()
                else:
                    # if RL_agent is not provided, the action is set to zeros because of the need of the parallel environment,
                    # but inside the backbone algorithm, the parameter controlled by action will use the default choices
                    action=np.zeros(opts.batch_size)
                
                # put action into environment(backbone algorithm to be specific)
                next_state,rewards,is_end,info = problem.step(action)
                state=next_state
                if agent:
                    state=torch.FloatTensor(state).to(opts.device)
                    
                # store the rollout cost history
                for tt in range(opts.batch_size):
                    cost_rollout[tt,t]+=info[tt]['gbest_val']
                if is_end.all():
                    if t+1<T:
                        for tt in range(opts.batch_size):
                            cost_rollout[tt,t+1:]+=info[tt]['gbest_val']
                    # store the final cost in the end of optimization process
                    for tt in range(opts.batch_size):
                        collect_gbest[tt,i]=info[tt]['gbest_val']
                    break
        # collect the mean and std of final cost
        collect_std.append(np.mean(np.std(collect_gbest,axis=-1)).item())
        collect_mean.append(np.mean(collect_gbest).item())
        # close the 
        problem.close()


    cost_rollout/=time_eval
    cost_rollout=np.mean(cost_rollout,axis=0)
    

    # save rollout data to file
    saving_path=os.path.join(opts.log_dir,opts.RL_agent,"{}_{}".format(opts.problem,
                                                        opts.dim),"rollout_{}_{}".format(opts.run_name,epoch_id))
    # only save part of the optimization process
    save_list=[cost_rollout[int((opts.dim**(k/5-3) * opts.max_fes )// opts.population_size -1 )].item() for k in range(15)]
    save_dict={'mean':np.mean(collect_mean).item(),'std':np.mean(collect_std).item(),'process':save_list}
    np.save(saving_path,save_dict)

    # log to tensorboard if needed
    if tb_logger:
        log_to_val(tb_logger,cost_rollout,epoch_id)
    
    # calculate and return the mean and std of final cost
    return np.mean(collect_gbest).item(),np.mean(collect_std).item()