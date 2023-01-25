import torch
import numpy as np
from utils.utils import set_seed
from scipy import integrate
import math
import gym
from gym import spaces
rand_seed=42

'''implementation of DMSPSO as an environment for DRL usage'''


class DMS_PSO_np(gym.Env):
    def __init__(self, dim=10, ps=100, max_fes=200000, max_x=100, c1=1.49445, c2=1.49445,  w=0.729,  cuda='cpu',
                        m=3,R=10,problem=None,reward_scale=100,max_velocity=10,
                        origin=False,boarder_method='clipping',reward_func='direct',origin_type='fixed',w_decay=True):
        super(DMS_PSO_np,self).__init__()
        self.dim,self.ps,self.max_fes,self.max_x=dim,ps,max_fes,max_x
        self.w,self.c1,self.c2=w,c1,c2
        self.m,self.R=m,R
        self.cuda=cuda
        self.n_swarm=self.ps//self.m
        self.fes=0
        self.problem=problem
        self.max_velocity=max_velocity
        self.origin,self.origin_type=origin,origin_type
        self.boarder_method=boarder_method
        self.reward_func=reward_func
        self.w_decay=w_decay
        
        self.group_index=np.zeros(ps,dtype=np.int8)
        self.per_no_improve=np.zeros(ps)
        self.lbest_no_improve=np.zeros(self.n_swarm)
        self.max_dist=np.sqrt((2*max_x)**2*dim)
        self.max_step=self.max_fes//self.ps
        assert ps%m==0, 'population cannot be update averagely'
        for sub_swarm in range(self.n_swarm):
            if sub_swarm!=self.n_swarm-1:
                self.group_index[sub_swarm*self.m:(sub_swarm+1)*self.m]=sub_swarm
            else:
                self.group_index[sub_swarm*self.m:]=sub_swarm
        self.node_dim=9
        self.reward_scale=reward_scale
        self.action_space=spaces.Box(low=0,high=1,shape=(self.ps,))
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(self.ps,27))

        # print(f'sDMS-PSO with {self.dim} dim.')

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

    # initilize the population
    def initilize(self):
        # randomly generate the positions and velocities
        rand_pos=np.random.uniform(low=-self.max_x,high=self.max_x,size=(self.ps,self.dim))
        rand_vel = np.random.uniform(low=-self.max_velocity,high=self.max_velocity,size=(self.ps,self.dim))
        
        # get the initial costs
        c_cost = self.get_costs(rand_pos) 

        # find out the gbest
        gbest_val = np.min(c_cost)
        gbest_index = np.argmin(c_cost)
        gbest_position=rand_pos[gbest_index]
        self.max_cost=np.min(c_cost)
        
        # store the population infomation
        self.particles={'current_position': rand_pos.copy(), #  ps, dim
                        'c_cost': c_cost.copy(), #  ps
                        'pbest_position': rand_pos.copy(), # ps, dim
                        'pbest': c_cost.copy(), #  ps
                        'gbest_position':gbest_position.copy(), # dim
                        'gbest_val':gbest_val,  # 1
                        'velocity': rand_vel.copy(), # ps,dim
                        }
        self.particles['lbest_cost']=np.zeros(self.n_swarm)
        self.particles['lbest_position']=np.zeros((self.n_swarm,self.dim))

        # the exploration and exploitation feature, they will be updated or generated in self.update_lbest()
        self.pbest_feature=self.input_feature_encoding()
        self.lbest_feature=np.zeros((self.n_swarm,self.node_dim))
        
        
    # interface for reset the environment
    def reset(self):
        if self.w_decay:
            self.w=0.9
        self.gen=0
        self.fes-=self.fes
        self.per_no_improve-=self.per_no_improve
        self.lbest_no_improve-=self.lbest_no_improve
        self.initilize()
        self.random_regroup()
        self.update_lbest(init=True)
        self.fes_eval=np.zeros_like(self.fes)
        return self.get_state(self.input_feature_encoding())

    # randomly regruop the population
    def random_regroup(self):
        regroup_index=torch.randperm(n=self.ps)
        self.lbest_no_improve-=self.lbest_no_improve
        self.regroup_index=regroup_index
        self.particles['current_position']=self.particles['current_position'][regroup_index] 
        self.particles['c_cost']= self.particles['c_cost'][regroup_index] 
        self.particles['pbest_position']=self.particles['pbest_position'][regroup_index] 
        self.particles['pbest']= self.particles['pbest'][regroup_index]
        self.particles['velocity']=self.particles['velocity'][regroup_index]
        self.per_no_improve=self.per_no_improve[regroup_index]
        self.pbest_feature=self.pbest_feature[regroup_index]

    # update lbest position, lbest cost
    def update_lbest(self,init=False):
        if init:
            # find the lbest position and lbest cost
            grouped_pbest=self.particles['pbest'].reshape(self.n_swarm,self.m)
            grouped_pbest_pos=self.particles['pbest_position'].reshape(self.n_swarm,self.m,self.dim)
            grouped_pbest_fea=self.pbest_feature.reshape(self.n_swarm,self.m,self.node_dim)
            self.particles['lbest_cost']=np.min(grouped_pbest,axis=-1)
            index=np.argmin(grouped_pbest,axis=-1)
            self.particles['lbest_position']=grouped_pbest_pos[range(self.n_swarm),index]

            # generate the lbest_feature(exploitation feature)
            self.lbest_feature=grouped_pbest_fea[range(self.n_swarm),index]
        else:
            # update the lbest position and lbest cost
            grouped_pbest=self.particles['pbest'].reshape(self.n_swarm,self.m)
            grouped_pbest_pos=self.particles['pbest_position'].reshape(self.n_swarm,self.m,self.dim)
            lbest_cur=np.min(grouped_pbest,axis=-1)
            index=np.argmin(grouped_pbest,axis=-1)
            lbest_pos_cur=grouped_pbest_pos[range(self.n_swarm),index]
            filter_lbest=lbest_cur<self.particles['lbest_cost']
            self.particles['lbest_cost']=np.where(filter_lbest,lbest_cur,self.particles['lbest_cost'])
            self.particles['lbest_position']=np.where(filter_lbest[:,None],lbest_pos_cur,self.particles['lbest_position'])
            self.lbest_no_improve=np.where(filter_lbest,np.zeros_like(self.lbest_no_improve),self.lbest_no_improve+1)

            # update the lbest feature(exploitation feature)
            if not self.origin:
                self.update_lbest_feature(filter_lbest)

    
    def get_pbest_lbest_feature(self):
        return (self.pbest_feature,self.lbest_feature[:,self.group_index])

    # return total state
    def get_state(self,cur_fea):
        state=np.concatenate((cur_fea,self.pbest_feature,self.lbest_feature[self.group_index]),axis=-1)
        return state

    # interface for population feature encoding
    def input_feature_encoding(self):
        max_step=self.max_fes//self.ps
        # cost cur
        fea0= self.particles['c_cost']/self.max_cost
        # cost cur_lbest
        fea1=(self.particles['c_cost']-self.particles['lbest_cost'][self.group_index])/self.max_cost     #  ps
        # cost cur_pbest
        fea2=(self.particles['c_cost']-self.particles['pbest'])/self.max_cost
        # fes cur_fes
        fea3=np.full(shape=(self.ps),fill_value=(self.max_fes-self.fes)/self.max_fes)
        # no_improve  per
        fea4=self.per_no_improve/max_step
        # no_improve  sub_group
        # fea5=np.full(shape=(self.ps) , fill_value=self.no_improve/max_step )
        fea5=self.lbest_no_improve[self.group_index]/self.R
        # dist cur_lbest
        # fea6=torch.sqrt(torch.sum((self.particles['current_position']-self.particles['gbest_position'].unsqueeze(1))**2,dim=-1))/self.max_dist
        fea6=np.sqrt(np.sum((self.particles['current_position']-self.particles['lbest_position'][self.group_index])**2,axis=-1))/self.max_dist
        # dist cur_pbest
        # fea7=torch.sqrt(torch.sum((self.particles['current_position']-self.particles['pbest_position'])**2,dim=-1))/self.max_dist
        fea7=np.sqrt(np.sum((self.particles['current_position']-self.particles['pbest_position'])**2,axis=-1))/self.max_dist
        '''angle state'''
        # bs, ps, dim
        pbest_cur_vec=self.particles['pbest_position']-self.particles['current_position']
        gbest_cur_vec=self.particles['lbest_position'][self.group_index]-self.particles['current_position']
        # cos angle
        fea8=np.sum(pbest_cur_vec*gbest_cur_vec,axis=-1)/((np.sqrt(np.sum(pbest_cur_vec**2,axis=-1))*np.sqrt(np.sum(gbest_cur_vec**2,axis=-1)))+1e-5)
        fea8=np.where(np.isnan(fea8),np.zeros_like(fea8),fea8)

        return np.concatenate((fea0[:,None],fea1[:,None],fea2[:,None],fea3[:,None],fea4[:,None],fea5[:,None],fea6[:,None],fea7[:,None],fea8[:,None]),axis=-1)

    
        
    # interface for pbest feature(exploration feature) updating
    def update_pbest_feature(self,filter):
        
        max_step=self.max_fes//self.ps
        # cost cur
        fea0= self.particles['pbest']/self.max_cost
        # cost cur_lbest
        fea1=(self.particles['pbest']-self.particles['lbest_cost'][self.group_index])/self.max_cost     # bs, ps
        # cost cur_pbest
        fea2=np.zeros_like(fea0)
        # fes cur_fes
        fea3=np.full(shape=(self.ps),fill_value=(self.max_fes-self.fes)/self.max_fes)
        fea3=np.where(filter,fea3,self.pbest_feature[:,3])
        # no_improve  per
        fea4=self.per_no_improve/max_step
        # no_improve  sub_group
        fea5=self.lbest_no_improve[self.group_index]/self.R
        # dist cur_lbest
        fea6=np.sqrt(np.sum((self.particles['pbest_position']-self.particles['lbest_position'][self.group_index])**2,axis=-1))/self.max_dist
        # dist cur_pbest
        fea7=np.zeros_like(fea0)
        '''angle state'''
        # bs, ps, dim
        fea8=np.zeros_like(fea0)

        self.pbest_feature=np.concatenate((fea0[:,None],fea1[:,None],fea2[:,None],fea3[:,None],fea4[:,None],fea5[:,None],fea6[:,None],fea7[:,None],fea8[:,None]),axis=-1)

    # inferface for lbest feature(exploitation feature) updating
    def update_lbest_feature(self,filter):
        max_step=self.max_fes//self.ps
        # cost cur
        fea0= self.particles['lbest_cost']/self.max_cost
        # cost cur_lbest
        fea1=np.zeros_like(fea0)     # bs, ps
        # cost cur_pbest
        fea2=np.zeros_like(fea0)
        # fes cur_fes
        fea3=np.full(shape=(self.n_swarm),fill_value=(self.max_fes-self.fes)/self.max_fes)
        fea3=np.where(filter,fea3,self.lbest_feature[:,3])
        # no_improve  per
        fea4=self.lbest_no_improve/max_step
        # no_improve  sub_group
        fea5=self.lbest_no_improve/self.R
        # dist cur_lbest
        fea6=np.zeros_like(fea0)
        # dist cur_pbest
        fea7=np.zeros_like(fea0)
        '''angle state'''
        # bs, ps, dim
        fea8=np.zeros_like(fea0)

        self.lbest_feature=np.concatenate((fea0[:,None],fea1[:,None],fea2[:,None],fea3[:,None],fea4[:,None],fea5[:,None],fea6[:,None],fea7[:,None],fea8[:,None]),axis=-1)

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


    # calculate reward according to the specified 
    def cal_reward(self,new_gbest,pre_gbest):
        if self.reward_func=='11':
            reward=self.cal_reward_11(new_gbest,pre_gbest)
        elif self.reward_func=='direct':
            reward=self.cal_reward_direct(new_gbest,pre_gbest)
        elif self.reward_func=='relative':
            reward=self.cal_reward_relative(new_gbest,pre_gbest)
        elif self.reward_func=='triangle':
            reward=self.cal_reward_triangle(new_gbest,pre_gbest)
        return reward*self.reward_scale

    def step(self,action=None):
        pre_gbest=self.particles['gbest_val']
        is_end=False
        self.gen+=1

        # linearly decreasing the coefficient of inertia w
        if self.w_decay:
            self.w-=0.5/(self.max_fes/self.ps)
        
        # if dmspso is not controlled by RL_agent, it will have two mode, one is local search, the other is the global search which is the same as GPSO
        # but if the DMSPSO is controlled by RL_agent, it will only have the local search mode
        cur_mode='ls'
        if self.fes>=0.9*self.max_fes and self.origin:
            cur_mode='gs'
        if self.origin:
            if self.origin_type=='fixed':
                c1=self.c1
                c2=self.c2
            elif self.origin_type=='normal':
                rand_mu=np.random.rand(self.ps)
                rand_std=np.random.rand(self.ps)*0.7
                action = np.random.normal(loc=rand_mu,scale=rand_std)
                action=np.clip(action,a_min=0,a_max=1)
                action=action[:,None]
                c_sum=self.c1+self.c2
                c1=action*c_sum
                c2=c_sum-c1
        else:
            # if the algorithm is controlled by RL agent, get parameters from actions
            c_sum=self.c1+self.c2
            action=action[:,None]
            c1=action*c_sum
            c2=c_sum-c1
        # update velocity
        rand1=np.random.rand(self.ps,1)
        rand2=np.random.rand(self.ps,1)
        v_pbest=rand1*(self.particles['pbest_position']-self.particles['current_position'])
        if cur_mode=='ls':
            v_lbest=rand2*(self.particles['lbest_position'][self.group_index]-self.particles['current_position'])
            new_velocity=self.w*self.particles['velocity']+c1*v_pbest+c2*v_lbest
        elif cur_mode=='gs':
            v_gbest=rand2*(self.particles['gbest_position'][None,:]-self.particles['current_position'])
            new_velocity=self.w*self.particles['velocity']+c1*v_pbest+c2*v_gbest
        new_velocity=np.clip(new_velocity,-self.max_velocity,self.max_velocity)

        # update position
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

        # get new cost
        new_cost=self.get_costs(new_position)

        # update particles
        filters = new_cost < self.particles['pbest']
        new_cbest_val = np.min(new_cost)
        new_cbest_index = np.argmin(new_cost)
        filters_best_val=new_cbest_val<self.particles['gbest_val']

        new_particles = {'current_position': new_position, # bs, ps, dim
                            'c_cost': new_cost, # bs, ps
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
                            'lbest_position':self.particles['lbest_position'],
                            'lbest_cost':self.particles['lbest_cost']
                            }

        # update the stag step for every single 
        filter_per_patience=new_particles['c_cost']<self.particles['c_cost']
        self.per_no_improve+=1
        tmp=np.where(filter_per_patience,self.per_no_improve,np.zeros_like(self.per_no_improve))
        self.per_no_improve-=tmp
        

        self.particles=new_particles

        # update pbest feature
        self.update_pbest_feature(filters)

        # update lbest-related information
        self.update_lbest()
        reward=self.cal_reward(self.particles['gbest_val'],pre_gbest)
        
        # regroup the population periodically
        if self.gen%self.R==0:
            self.random_regroup()
            self.update_lbest(init=True)
        
        # see if the end condition is satisfied
        if self.fes>=self.max_fes:
            is_end=True
        if self.particles['gbest_val']<=1e-8:
            is_end=True
        
        # update state
        next_state=self.input_feature_encoding()
        # next_state=self.observe(next_state)
        next_state=self.get_state(self.input_feature_encoding())
        info={'gbest_val':self.particles['gbest_val'],'fes_used':self.fes_eval}
        
        
        return (next_state,reward ,is_end,info)

