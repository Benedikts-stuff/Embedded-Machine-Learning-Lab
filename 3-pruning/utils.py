from typing import List, Dict, Tuple

import torch
from torchinfo import summary
import time
import io
import gzip

import matplotlib.pyplot as plt
import numpy as np
import copy

def net_macs(model_class: torch.nn.Module, state_dict: Dict) -> int:
    net = model_class()
    net.load_state_dict(state_dict)
    res = summary(net, (1, 3, 32, 32), verbose=0)
    return res.total_mult_adds

def net_params(model_class: torch.nn.Module, state_dict: Dict) -> int:
    net = model_class()
    net.load_state_dict(state_dict)
    res = summary(net, (1, 3, 32, 32), verbose=0)
    return res.total_params

def net_time(model_class: torch.nn.Module, state_dict: Dict,
             iterations: int=5, device: str='cpu') -> float:

    torch_device = torch.device(device)
    net = model_class()
    net.load_state_dict(state_dict)
    net.eval()
    
    t = 0.0
    
    input = torch.rand(64, 3, 32, 32)
    input = input.to(torch_device)
    
    for _ in range(10):
        net.to(torch_device)
        torch.cuda.synchronize()
        out = net(input)
        torch.cuda.synchronize()
        torch.max(out)
    times = []
    for _ in range(iterations):
        t_start = time.time()
        torch.cuda.synchronize()
        out = net(input)
        torch.cuda.synchronize()
        t_end = time.time()
        torch.max(out)
        times.append(t_end - t_start)
        
    times = list(sorted(times, reverse=False))
        
    return np.mean(times[0:10])
    
def net_acc(model_class: torch.nn.Module,
            state_dict: Dict,
            testloader: torch.utils.data.DataLoader,
            device: str='cpu', batches: int=10) -> float:

    net = model_class()
    torch_device = torch.device(device)
    
    net.load_state_dict(state_dict)
    net.to(torch_device)
    correct_predictions = 0
    total_predictions = 0
    
    for idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(torch_device)
        outputs = net(inputs)
        
        correct_predictions +=(torch.argmax(outputs.cpu().detach(), axis=1) == 
                               targets.cpu().detach()).sum()
        
        total_predictions += int(targets.shape[0])
        
        if idx == (batches - 1):
            break
            
    accuracy = float(correct_predictions/total_predictions)
    return round(100*accuracy, 2)

def size_on_disk(state_dict: Dict) -> Tuple[int, int]:
    buff = io.BytesIO()
    torch.save(state_dict, buff)
    compressed_buff = gzip.compress(buff.getvalue(), compresslevel=9)
    return compressed_buff.__sizeof__(), buff.__sizeof__()


def plot(data, xlabel='Execution time', save_path='plot.png'):
   
    data = copy.deepcopy(data)
    for (x, y, label) in data:
        x = np.array(x)/max(x)
        plt.plot(x, y, label=label, alpha=0.5)
        plt.scatter(x, y, alpha=0.5)
        
    plt.ylabel('Accuracy')
    plt.xlabel(xlabel)
    plt.legend()
        
    plt.savefig(save_path)
    plt.show()