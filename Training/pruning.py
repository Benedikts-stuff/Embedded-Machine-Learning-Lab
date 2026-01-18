import torch
import torch.nn as nn
import numpy as np

def get_l1_pruning_mask(weight, ratio):
    l1_norm = weight.abs().sum(dim=(1, 2, 3))
    num_filters = l1_norm.shape[0]
    num_keep = int(num_filters * (1 - ratio))
    
    _, keep_indices = torch.topk(l1_norm, num_keep, sorted=False)
    return sorted(keep_indices.tolist())

def apply_structured_pruning(state_dict, ratio):
    pruned_sd = state_dict.copy()
    masks = {}

    for i in range(1, 9):
        w_key = f'conv{i}.weight'
        masks[i] = get_l1_pruning_mask(state_dict[w_key], ratio)
        
    return masks