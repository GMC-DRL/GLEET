from asyncio import base_tasks
from turtle import st
# import torch
import copy

from .cec_dataset import *

from utils.utils import set_seed
import numpy as np
import gym
from gym import spaces
from scipy import integrate
rand_seed=42
reward_threshold=1e-7


'''implementation of GPSO as an environment for DRL usage'''


# inherit gym.Env for the convenience of parallelism
class GPSO_numpy(gym.Env):
    def __init__(self,dim=30,ps=100,w_decay=True,c=4.1, max_velocity = 10,problem=None,
                 max_x=100, reward_scale = 100,max_fes=100000,origin=False,
                 boarder_method='clipping',reward_func='direct',origin_type='fixed'):
        super(GPSO_numpy,self).__init__()
        
        self.dim = dim
        self.w_decay=w_decay
        if self.w_decay:
            self.w=0.9
        else:
            self.w=0.729
        self.c=c
        self.max_velocity = max_velocity
        self.max_x=max_x
        self.reward_scale = reward_scale
        self.origin=origin
        self.origin_type=origin_type
        self.ps=ps

        # instance
        self.problem=problem
        
        self.no_improve=0
        self.per_no_improve=np.zeros((self.ps,))
        self.fes=0
        self.evaling=False
        self.max_fes=max_fes
        self.max_dist=np.sqrt((2*max_x)**2*dim)
        
        self.boarder_method=boarder_method
        
        self.reward_func=reward_func
        
        
        self.action_space=spaces.Box(low=0,high=1,shape=(self.ps,))
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(self.ps,9))
        self.name='GPSO'
        # print(f'GPSO with {self.dim} dims .')


    # initialize GPSO environment
    def initialize_particles(self):
        # randomly generate the position and velocity
        rand_pos=np.random.uniform(low=-self.max_x,high=self.max_x,size=(self.ps,self.dim))
        rand_vel = np.random.uniform(low=-self.max_velocity,high=self.max_velocity,size=(self.ps,self.dim))
        
        # get the initial cost
        c_cost = self.get_costs(rand_pos) # ps

        # find out the gbest_val
        gbest_val = np.min(c_cost)
        gbest_index = np.argmin(c_cost)
        gbest_position=rand_pos[gbest_index]

        # record
        self.max_cost=np.min(c_cost)

        # store all the information of the paraticles
        self.particles={'current_position': rand_pos.copy(), #  ps, dim
                        'c_cost': c_cost.copy(), #  ps
                        'pbest_position': rand_pos.copy(), # ps, dim
                        'pbest': c_cost.copy(), #  ps
                        'gbest_position':gbest_position.copy(), # dim
                        'gbest_val':gbest_val,  # 1
                        'velocity': rand_vel.copy(), # ps,dim
                        'gbest_index':gbest_index # 1
                        }

    # the interface for environment reseting
    def reset(self):
        # set the hyperparameters back to init value if needed
        if self.w_decay:
            self.w=0.9
            
        self.no_improve-=self.no_improve
        self.fes-=self.fes
        self.per_no_improve-=self.per_no_improve
        
        # initialize the population
        self.initialize_particles()
        
        # get the population state
        state=self.observe() # ps, 9

        # get the exploration state
        self.pbest_feature=state.copy()    # ps, 9

        # get the explotation state
        self.gbest_feature=state[self.particles['gbest_index']] # 9
        
        # get and return the total state (population state, exploration state, exploitation state)
        gp_cat=self.gp_cat()  # ps, 18
        return np.concatenate((state,gp_cat),axis=-1)   # ps, 9+18

    def eval(self):
        set_seed(rand_seed)
        self.evaling=True

    def train(self):
        set_seed()
        self.evaling=False


    # calculate costs of solutions
    def get_costs(self,position):
        ps=position.shape[0]
        self.fes+=ps
        cost=self.problem.func(position)
        return cost
    
    # feature encoding
    def observe(self):
        max_step=self.max_fes//self.ps
        # cost cur
        fea0= self.particles['c_cost']/self.max_cost
        # cost cur_gbest
        fea1=(self.particles['c_cost']-self.particles['gbest_val'])/self.max_cost     #  ps
        # cost cur_pbest
        fea2=(self.particles['c_cost']-self.particles['pbest'])/self.max_cost
        # fes cur_fes
        fea3=np.full(shape=(self.ps),fill_value=(self.max_fes-self.fes)/self.max_fes)
        # no_improve  per
        fea4=self.per_no_improve/max_step
        # no_improve  whole
        fea5=np.full(shape=(self.ps) , fill_value=self.no_improve/max_step )
        # distance between cur and gbest
        fea6=np.sqrt(np.sum((self.particles['current_position']-np.expand_dims(self.particles['gbest_position'],axis=0))**2,axis=-1))/self.max_dist
        # distance between cur and pbest
        fea7=np.sqrt(np.sum((self.particles['current_position']-self.particles['pbest_position'])**2,axis=-1))/self.max_dist
        
        # cos angle
        pbest_cur_vec=self.particles['pbest_position']-self.particles['current_position']
        gbest_cur_vec=np.expand_dims(self.particles['gbest_position'],axis=0)-self.particles['current_position']
        fea8=np.sum(pbest_cur_vec*gbest_cur_vec,axis=-1)/((np.sqrt(np.sum(pbest_cur_vec**2,axis=-1))*np.sqrt(np.sum(gbest_cur_vec**2,axis=-1)))+1e-5)
        fea8=np.where(np.isnan(fea8),np.zeros_like(fea8),fea8)

        return np.concatenate((fea0[:,None],fea1[:,None],fea2[:,None],fea3[:,None],fea4[:,None],fea5[:,None],fea6[:,None],fea7[:,None],fea8[:,None]),axis=-1)

    def gp_cat(self):
        return np.concatenate((self.pbest_feature,self.gbest_feature[None,:].repeat(self.ps,axis=0)),axis=-1)   # ps, 18
        
    # direct reward function
    def cal_reward_direct(self,new_gbest,pre_gbest):
        bonus_reward=(pre_gbest-new_gbest)/self.max_cost
        assert np.min(bonus_reward)>=0,'reward should be bigger than 0!'
        return bonus_reward

    # 1 -1 reward function
    def cal_reward_11(self,new_gbest,pre_gbest):
        if new_gbest<pre_gbest:
            reward=1
        else:
            reward=-1
        return reward

    # relative reward function
    def cal_reward_relative(self,new_gbest,pre_gbest):
        return (pre_gbest-new_gbest)/pre_gbest

    # triangle reward function
    def cal_reward_triangle(self,new_gbest,pre_gbest):
        reward=0
        if new_gbest<pre_gbest:
            p_t=(self.max_cost-pre_gbest)/self.max_cost
            p_t_new=(self.max_cost-new_gbest)/self.max_cost
            reward=0.5*(p_t_new**2-p_t**2)
        else:
            reward=0
        assert reward>=0,'reward should be bigger than 0!'
        return reward

    


    def seed(self, seed=None):
        np.random.seed(seed)

    def step(self, action=None):
        is_end=False
        
        # record the gbest_val in the begining
        pre_gbest=self.particles['gbest_val']

        # linearly decreasing the coefficient of inertia w
        if self.w_decay:
            self.w-=0.5/(self.max_fes/self.ps)

        # generate two set of random val for pso velocity update
        rand1=np.random.rand(self.ps,1)
        rand2=np.random.rand(self.ps,1)
        
        # if the algorithm isn't controlled by RL_agent, generate the default action
        if self.origin:
            if self.origin_type=='fixed':
                action=np.ones(self.ps) * 0.5
            elif self.origin_type=='nofixed':
                action = np.random.rand(self.ps)
            elif self.origin_type=='normal':
                rand_mu=np.random.rand(self.ps)
                rand_std=np.random.rand(self.ps)
                action = np.random.normal(loc=rand_mu,scale=rand_std)
                action=np.clip(action,a_min=0,a_max=1)

        action = action[:,None]

        # update velocity
        new_velocity = self.w*self.particles['velocity']+self.c*action*rand1*(self.particles['pbest_position']-self.particles['current_position'])+ \
                        self.c*(1-action)*rand2*(self.particles['gbest_position'][None,:]-self.particles['current_position'])

        # clip the velocity if exceeding the boarder
        new_velocity=np.clip(new_velocity,-self.max_velocity,self.max_velocity)

        # update position according the boarding method
        if self.boarder_method=="clipping":
            raw_position=self.particles['current_position']+new_velocity
            new_position = np.clip(raw_position,-self.max_x,self.max_x)
        elif self.boarder_method=="random":
            raw_position=self.particles['current_position']+new_velocity
            filter=raw_position.abs()>self.max_x
            new_position=np.where(filter,np.random.uniform(low=-self.max_x,high=self.max_x,size=(self.ps,self.dim)),raw_position)
        elif self.boarder_method=="periodic":
            raw_position=self.particles['current_position']+new_velocity
            new_position=-self.max_x+((raw_position-self.max_x)%(2.*self.max_x))
        elif self.boarder_method=="reflect":
            raw_position=self.particles['current_position']+new_velocity
            filter_low=raw_position<-self.max_x
            filter_high=raw_position>self.max_x
            new_position=np.where(filter_low,-self.max_x+(-self.max_x-raw_position),raw_position)
            new_position=np.where(filter_high,self.max_x-(new_position-self.max_x),new_position)


        # calculate the new costs
        new_cost = self.get_costs(new_position)

        # update particles
        filters = new_cost < self.particles['pbest']
        
        new_cbest_val = np.min(new_cost)
        new_cbest_index = np.argmin(new_cost)
        filters_best_val=new_cbest_val<self.particles['gbest_val']
        
        new_particles = {'current_position': new_position, 
                            'c_cost': new_cost, 
                            'pbest_position': np.where(np.expand_dims(filters,axis=-1),
                                                        new_position,
                                                        self.particles['pbest_position']),
                            'pbest': np.where(filters,
                                                new_cost,
                                                self.particles['pbest']),
                            'velocity': new_velocity,
                            'gbest_val':np.where(filters_best_val,
                                                    new_cbest_val,
                                                    self.particles['gbest_val']),
                            'gbest_position':np.where(np.expand_dims(filters_best_val,axis=-1),
                                                        new_position[new_cbest_index],
                                                        self.particles['gbest_position']),
                            'gbest_index':np.where(filters_best_val,new_cbest_index,self.particles['gbest_index'])
                            }

        # update the stagnation steps for the whole population
        if new_particles['gbest_val']<self.particles['gbest_val']:
            self.no_improve=0
        else:
            self.no_improve+=1
        
        # update the stagnation steps for singal particle in the population
        filter_per_patience=new_particles['c_cost']<self.particles['c_cost']
        self.per_no_improve+=1
        tmp=np.where(filter_per_patience,self.per_no_improve,np.zeros_like(self.per_no_improve))
        self.per_no_improve-=tmp
        
        # update the population
        self.particles=new_particles

        # see if the end condition is satisfied
        if self.fes>=self.max_fes:
            is_end=True
        if self.particles['gbest_val']<=1e-8:
            is_end=True
        
        # cal the reward
        if self.reward_func=='11':
            reward=self.cal_reward_11(self.particles['gbest_val'],pre_gbest)
        elif self.reward_func=='direct':
            reward=self.cal_reward_direct(self.particles['gbest_val'],pre_gbest)
        elif self.reward_func=='relative':
            reward=self.cal_reward_relative(self.particles['gbest_val'],pre_gbest)
        elif self.reward_func=='triangle':
            reward=self.cal_reward_triangle(self.particles['gbest_val'],pre_gbest)
        reward*=self.reward_scale
        
        # get next state
        next_state=self.observe()
        # update exploration state
        self.pbest_feature=np.where(self.per_no_improve[:,None]==0,next_state,self.pbest_feature)
        # update exploitation state
        if self.no_improve==0:
            self.gbest_feature=next_state[self.particles['gbest_index']]
        next_gpcat=self.gp_cat()
        info={'gbest_val':self.particles['gbest_val']}
        return (np.concatenate((next_state,next_gpcat),axis=-1),reward,is_end,info)

