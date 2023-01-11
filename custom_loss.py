import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def get_device(target_tensor):
    try:
        device = target_tensor.device
        #print(device)
        return target_tensor.device
    except:
        return 'cpu'

def create_weights(class_weights, targets):
    weights = np.zeros(targets.shape)
    for ind in range(targets.shape[1]):
        a = targets[:, ind]
        weights[:, ind] = np.where(a==0,
		 class_weights['negative_weights'][ind],
		 class_weights['positive_weights'][ind])
    return weights

class WeightedBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(WeightedBCELoss, self).__init__()

    def forward(self, inputs, targets, class_weights):
        
        #first compute binary cross-entropy 
        weights = create_weights(class_weights,
                    targets.cpu().numpy(),
                    )
        device = get_device(targets)
        weights = torch.from_numpy(weights).to(device)
        loss = F.binary_cross_entropy(inputs, targets, weight=weights, reduction='mean')
                       
        return loss
