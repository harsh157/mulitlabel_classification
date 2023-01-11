

import torch
from torch import nn
import torch.nn.functional as F

class NNModel(nn.Module):
    
    def __init__(self, input_size, hidden_dims, num_classes, dropout=0.1):
        
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.output_layer = nn.Linear(hidden_dims[1], num_classes)

        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, x):

        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        logits = self.output_layer(x)
        #print(logits.shape)
        #logits = logits.squeeze(-2)
        #print(logits.shape)
        return logits
