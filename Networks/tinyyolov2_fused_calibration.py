import torch
import torch.nn as nn
import torch.nn.functional as F
from Training.quantization import tensor_scale_asym

class TinyYoloV2Calibration(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.calib_dict = {}

    def forward(self, x):
        self.calib_dict["input_scale"], _ = tensor_scale_asym(x)
        
        for i in range(1, 9):
            layer = getattr(self.model, f"conv{i}")
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            self.calib_dict[f"conv{i}_out_scale"], self.calib_dict[f"conv{i}_zp"] = tensor_scale_asym(x)
            
            if i <= 5: x = F.max_pool2d(x, 2, 2)
            elif i == 6: 
                x = F.pad(x, (0, 1, 0, 1))
                x = F.max_pool2d(x, 2, 1)

        x = self.model.conv9(x)
        self.calib_dict["output_scale"], _ = tensor_scale_asym(x)
        return x