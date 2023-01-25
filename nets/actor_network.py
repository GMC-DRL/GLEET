from torch import nn
import torch
from nets.graph_layers import MultiHeadEncoder, MLP, EmbeddingNet, MultiHeadCompat
from torch.distributions import Normal
import torch.nn.functional as F

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class Actor(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_heads_actor,
                 n_heads_decoder,
                 n_layers,
                 normalization,
                 v_range,
                 node_dim,
                 hidden_dim1,
                 hidden_dim2,
                 no_attn=False,
                 no_eef=False,
                 max_sigma=0.7,
                 min_sigma=1e-3,
                 ):
        super(Actor, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads_actor = n_heads_actor
        self.n_heads_decoder = n_heads_decoder        
        self.n_layers = n_layers
        self.normalization = normalization
        self.range = v_range
        self.no_attn=no_attn
        self.no_eef=no_eef
        self.node_dim = node_dim 

        # figure out the Actor network
        if not self.no_attn:
            # figure out the embedder for feature embedding
            self.embedder = EmbeddingNet(
                                self.node_dim,
                                self.embedding_dim)
            # figure out the fully informed encoder
            self.encoder = mySequential(*(
                    MultiHeadEncoder(self.n_heads_actor,
                                    self.embedding_dim,
                                    self.hidden_dim,
                                    self.normalization,)
                for _ in range(self.n_layers))) # stack L layers

            # w/o eef for ablation study
            if not self.no_eef:
                # figure out the embedder for exploration and exploitation feature
                self.embedder_for_decoder = EmbeddingNet(2*self.embedding_dim, self.embedding_dim)
                # figure out the exploration and exploitation decoder
                self.decoder = mySequential(*(
                        MultiHeadEncoder(self.n_heads_actor,
                                        self.embedding_dim,
                                        self.hidden_dim,
                                        self.normalization,)
                    for _ in range(self.n_layers))) # stack L layers
            # figure out the mu_net and sigma_net
            self.mu_net = MLP(self.embedding_dim ,hidden_dim1,hidden_dim2, 1, 0) 
            self.sigma_net=MLP(self.embedding_dim,hidden_dim1,hidden_dim2,1,0)
        else:
            # w/o both
            if self.no_eef:
                self.mu_net=MLP(self.node_dim,16,8,1)
                self.sigma_net=MLP(self.node_dim,16,8,1,0)
            # w/o attn
            else:
                self.mu_net=MLP(3*self.node_dim,16,8,1)
                self.sigma_net=MLP(3*self.node_dim,16,8,1,0)

        self.max_sigma=max_sigma
        self.min_sigma=min_sigma
        
        print(self.get_parameter_number())

    def get_parameter_number(self):
        
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, x_in,fixed_action = None, require_entropy = False,to_critic=False,only_critic=False):
        if not self.no_attn:
            population_feature=x_in[:,:,:self.node_dim]
            eef=x_in[:,:,self.node_dim:]
            # pass through embedder
            h_em = self.embedder(population_feature)
            # pass through encoder
            logits = self.encoder(h_em)
            if not self.no_eef:
                # pass through the embedder to get eef embedding
                exploration_feature=eef[:,:,:9]
                exploitation_feature=eef[:,:,9:]
                exploration_eb=self.embedder(exploration_feature)   
                exploitation_eb=self.embedder(exploitation_feature)   
                x_in_decoder=torch.cat((exploration_eb,exploitation_eb),dim=-1)
                # pass through the embedder for decoder
                x_in_decoder = self.embedder_for_decoder(x_in_decoder)

                # pass through decoder
                logits = self.decoder(logits,x_in_decoder)
            # share logits to critic net, where logits is from the decoder output 
            if only_critic:
                return logits  # .view(bs, dim, ps, -1)
            # finally decide the mu and sigma
            mu = (torch.tanh(self.mu_net(logits))+1.)/2.
            sigma=(torch.tanh(self.sigma_net(logits))+1.)/2. * (self.max_sigma-self.min_sigma)+self.min_sigma
        else:
            feature=x_in
            if self.no_eef:
                feature=x_in[:,:,:self.node_dim]
            if only_critic:
                return feature
            mu = (torch.tanh(self.mu_net(feature))+1.)/2.
            sigma=(torch.tanh(self.sigma_net(feature))+1.)/2. * (self.max_sigma-self.min_sigma)+self.min_sigma

        # don't share the network between actor and critic if there is no attention mechanism
        _to_critic=feature if self.no_attn else logits

        policy = Normal(mu, sigma)
        

        if fixed_action is not None:
            action = torch.tensor(fixed_action)
        else:
            # clip the action to (0,1)
            action=torch.clamp(policy.sample(),min=0,max=1)
        # get log probability
        log_prob=policy.log_prob(action)

        # The log_prob of each instance is summed up, since it is a joint action for a population
        log_prob=torch.sum(log_prob,dim=1)

        
        if require_entropy:
            entropy = policy.entropy() # for logging only 
            
            out = (action,
                   log_prob,
                   _to_critic if to_critic else None,
                   entropy)
        else:
            out = (action,
                   log_prob,
                   _to_critic if to_critic else None,
                   )
        return out