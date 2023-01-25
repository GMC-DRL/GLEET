import torch
from torch import nn
from nets.graph_layers import MLP
# from nets.graph_layers import ValueDecoder


class Critic(nn.Module):

    def __init__(self,
             input_dim,
             hidden_dim1,
             hidden_dim2
             ):
        
        super(Critic, self).__init__()
        self.input_dim = input_dim
        # for GLEET, hidden_dim1 = 32, hidden_dim2 = 16
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        self.value_head=MLP(input_dim=self.input_dim,mid_dim1=hidden_dim1,mid_dim2=hidden_dim2,output_dim=1)

    def forward(self, h_features):
        # since it's joint actions, the input should be meaned at population-dimention
        h_features=torch.mean(h_features,dim=-2)
        # pass through value_head to get baseline_value
        baseline_value = self.value_head(h_features)    
        
        return baseline_value.detach().squeeze(), baseline_value.squeeze()
        
