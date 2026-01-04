import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import pdb

class WeightMat(nn.Module):
    def __init__(self, reduced_rew_dim, ori_rew_dim, device): # 4, 16
        super().__init__()

        # Define a learnable tensor as a class parameter
        self.learnable_matrix = nn.Parameter(torch.empty(reduced_rew_dim, ori_rew_dim, device=device))
        init.constant_(self.learnable_matrix, 1 / ori_rew_dim)

    def forward(self, input_tensor):
        # input_data: (batch, ori_rew_dim) torch.float32. output: (batch, reduced_rew_dim)  torch.float32.
        with torch.no_grad():
            output = F.linear(input_tensor, self.learnable_matrix)
        return output

    # def forward(self, input_data): # numpy - torch - numpy version
    #     # input_data: (ori_rew_dim,). output: (reduced_rew_dim,)
    #     # input_data: 'float64'
    #     input_tensor = torch.tensor(input_data).float() # torch.float32; torch.tensor to separate memory.
    #
    #     with torch.no_grad():
    #         output_tensor = torch.matmul(self.learnable_matrix, input_tensor) # torch.float32
    #
    #     output_numpy = output_tensor.detach().double().numpy() # 'float64'
    #     return output_numpy

