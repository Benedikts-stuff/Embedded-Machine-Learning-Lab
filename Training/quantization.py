import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.ao.nn.quantized.functional as F_q
import tqdm
import numpy as np

def tensor_scale(x):
    max_val = float(x.abs().max())
    return max_val / 127.0 if max_val > 0 else 1.0


def tensor_scale_asym(x):
    x_min = float(x.min())
    x_max = float(x.max())
    # Zurück zu quint8 Bereich: 0 bis 255
    qmin, qmax = 0, 255 
    if x_max == x_min: return 1.0, 0
    scale = (x_max - x_min) / (qmax - qmin)
    # Zero Point Berechnung für Unsigned
    zp = round(qmin - x_min / scale)
    # Clamp auf den Bereich 0-255 (sehr wichtig für quint8!)
    zp = max(qmin, min(qmax, zp))
    return scale, int(zp)


class QConv2dLeakyReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, use_relu=True):
        super().__init__()
        # Dummy-Initialisierung
        qweight = torch.quantize_per_tensor(torch.randn(out_ch, in_ch, kernel_size, kernel_size), 
                                            scale=0.01, zero_point=0, dtype=torch.qint8)
        self.weight = nn.Parameter(qweight, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_ch), requires_grad=False)
        
        # Getrennte Buffer für Conv-Output und ReLU-Output
        self.register_buffer('conv_scale', torch.tensor(0.1))
        self.register_buffer('conv_zp', torch.tensor(0))
        self.register_buffer('relu_scale', torch.tensor(0.1))
        self.register_buffer('relu_zp', torch.tensor(0))
        
        self.stride, self.padding, self.use_relu = stride, padding, use_relu
        self.repack_weights()

    def repack_weights(self):
        self._prepack = torch.ops.quantized.conv2d_prepack(
            self.weight, 
            self.bias, 
            [self.stride, self.stride], 
            [self.padding, self.padding], # Padding (Pos 4)
            [1, 1],                       # Dilation (Pos 5)
            1                             # Groups
        )

    def forward(self, x):
        # 1. Conv mit Conv-Skalierung
        x = torch.ops.quantized.conv2d(x, self._prepack, self.conv_scale, self.conv_zp)
        # 2. Leaky ReLU mit eigener ReLU-Skalierung
        if self.use_relu:
            return F_q.leaky_relu(x, negative_slope=0.1, scale=float(self.relu_scale), zero_point=int(self.relu_zp))
        return x


def calibrate_model(model, loader, device, num_batches=12):
    model.eval()
    obs = {f"conv{i}_raw": [] for i in range(1, 10)}
    obs.update({f"conv{i}_relu": [] for i in range(1, 9)})
    obs["input"] = []

    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm.tqdm(loader)):
            if i >= num_batches: break
            images = images.to(device)
            obs["input"].append((images.min().item(), images.max().item()))
            
            x = images
            for idx in range(1, 10):
                x = getattr(model, f"conv{idx}")(x)
                # 1. Speichere Conv-Output (Wichtig für conv_scale)
                obs[f"conv{idx}_raw"].append((x.min().item(), x.max().item()))
                
                if idx < 9: 
                    x = torch.nn.functional.leaky_relu(x, 0.1)
                    # 2. HIER FEHLTE DIE ZEILE: Speichere ReLU-Output (Wichtig für relu_scale)
                    obs[f"conv{idx}_relu"].append((x.min().item(), x.max().item()))
                
                if idx <= 5: 
                    x = torch.nn.functional.max_pool2d(x, 2, 2)
                elif idx == 6: 
                    x = torch.nn.functional.max_pool2d(torch.nn.functional.pad(x, (0, 1, 0, 1)), 2, 1)

    calib_dict = {}
    for key, val in obs.items():
        if not val: continue
# Nimm den Mittelwert der Maxima statt des absoluten Maximums
        global_max = np.mean([v[1] for v in val]) * 1.1 # 10% Puffer
        global_min = np.mean([v[0] for v in val]) * 1.1
        
        dummy = torch.tensor([global_min, global_max])
        s, zp = tensor_scale_asym(dummy)
        
        if key == "input":
            calib_dict["input_scale"], calib_dict["input_zp"] = s, zp
        else:
            # Das erzeugt Keys wie "conv1_raw_scale" und "conv1_relu_scale"
            calib_dict[f"{key}_scale"], calib_dict[f"{key}_zp"] = s, zp
            
    return calib_dict
