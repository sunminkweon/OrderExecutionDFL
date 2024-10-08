import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleFeedFoward(nn.Module):
    """
    Just SimpleFeedFoward
    """
    def __init__(self, configs):
        super(SimpleFeedFoward, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_layers = configs.num_layers
        self.hidden_dim = configs.hidden_dim

        self.input_layers = nn.Linear(self.seq_len, self.hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU()
            ))

        self.output_layer = nn.Linear(self.hidden_dim, self.pred_len)
                               
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.input_layers(x.permute(0,2,1))
        
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        output = self.output_layer(x)
        
        return output.permute(0,2,1) # [Batch, Output length, Channel]