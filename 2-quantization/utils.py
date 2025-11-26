import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def correct_predictions(outputs, targets):
    correct_predictions = 0
    for i in range(len(targets)):
        correct_predictions += int(torch.argmax(outputs[i]) == targets[i])
    return correct_predictions


def net_time(model_class, testloader):
    # ----to-be-done-by-student-------------------
    net = model_class()
    batch, labels = next(iter(testloader))
    t_now = time.time()
    forward = net(batch)
    t_after = time.time()
    # ----to-be-done-by-student-------------------
    t = abs(t_now - t_after)
    return t


def net_acc(model_class, state_dict, testloader):
    # ----to-be-done-by-student-------------------
    net = model_class()
    net.load_state_dict(state_dict)
    batch, labels = next(iter(testloader))
    forward = net(batch)
    num_correct = correct_predictions(forward, labels)
    num_samples = len(labels)
    # ----to-be-done-by-student-------------------
    accuracy = num_correct / num_samples
    return accuracy

def tensor_scale(input):
    return float(torch.abs(torch.max(input))) / 127.0

def asym(input):
    return float(torch.max(input)/255.0)



def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_w, bn_b):
    """
    Input:
        conv_w: shape=(output_channels, in_channels, kernel_size, kernel_size)
        conv_b: shape=(output_channels)
        bn_rm:  shape=(output_channels)
        bn_rv:  shape=(output_channels)
        bn_w:   shape=(output_channels)
        bn_b:   shape=(output_channels)

    Output:
        fused_conv_w = shape=conv_w
        fused_conv_b = shape=conv_b
    """
    bn_eps = 1e-05

    fused_conv = torch.zeros(conv_w.shape)
    fused_bias = torch.zeros(conv_b.shape)

    # to-be-done-by-student
    print(bn_w.shape)
    print(bn_rv.shape)
    print(conv_w.shape)
    fused_conv = (bn_w / torch.sqrt(bn_rv + bn_eps)).reshape(-1, 1, 1, 1) * conv_w
    fused_bias = (bn_w * (conv_b - bn_rm) / torch.sqrt(bn_rv + bn_eps) + bn_b)
    # to-be-done-by-student

    return fused_conv, fused_bias


class QConv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(QConv2dReLU, self).__init__()

        self.weight = torch.nn.Parameter(torch.quantize_per_tensor(torch.Tensor(
            out_channels, in_channels // 1, *(kernel_size, kernel_size)), scale=0.1, zero_point=0, dtype=torch.qint8),
            requires_grad=False)
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels), requires_grad=False)

        self.register_buffer('scale', torch.tensor(0.1))

        self.stride = stride
        self.padding = padding

        self._prepack = self._prepare_prepack(self.weight, self.bias, stride, padding)
        self._register_load_state_dict_pre_hook(self._sd_hook)

    def _prepare_prepack(self, qweight, bias, stride, padding):
        assert qweight.is_quantized, "QConv2dReLU requires a quantized weight."
        assert not bias.is_quantized, "QConv2dReLU requires a float bias."
        return torch.ops.quantized.conv2d_prepack(qweight, bias, stride=[stride, stride], dilation=[1, 1],
                                                  padding=[padding, padding], groups=1)

    def _sd_hook(self, state_dict, prefix, *_):
        self._prepack = self._prepare_prepack(f_sd(state_dict, prefix + 'weight'), f_sd(state_dict, prefix + 'bias'),
                                              self.stride, self.padding)

    def forward(self, x):
        return torch.ops.quantized.conv2d_relu(x, self._prepack, self.scale, 64)

def f_sd(sd, endswith_key_string):
    keys = [i for i in sd.keys() if i.endswith(endswith_key_string)]
    if not keys:
        raise KeyError(endswith_key_string)
    return sd[keys[0]]

# Quantized Linear Module
class QLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(QLinear, self).__init__()

        self.weight = torch.nn.Parameter(
            torch.quantize_per_tensor(torch.Tensor(out_features, in_features), scale=0.1, zero_point=0,
                                      dtype=torch.qint8), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))

        self.register_buffer('scale', torch.tensor(0.1))

        self._prepack = self._prepare_prepack(self.weight, self.bias)

        self._register_load_state_dict_pre_hook(self._sd_hook)

    def _prepare_prepack(self, qweight, bias):
        assert qweight.is_quantized, "QConv2dReLU requires a quantized weight."
        assert not bias.is_quantized, "QConv2dReLU requires a float bias."
        return torch.ops.quantized.linear_prepack(qweight, bias)

    def _sd_hook(self, state_dict, prefix, *_):
        self._prepack = self._prepare_prepack(f_sd(state_dict, prefix + 'weight'), f_sd(state_dict, prefix + 'bias'))
        return

    def forward(self, x):
        return torch.ops.quantized.linear(x, self._prepack, self.scale, 64)


class QCifarNet(nn.Module):
    def __init__(self):
        super(QCifarNet, self).__init__()

        self.register_buffer("scale", torch.tensor(0.1))

        self.conv1 = QConv2dReLU(3, 16, 3, 1, padding=1)
        self.conv2 = QConv2dReLU(16, 16, 3, 1, padding=1)

        self.conv3 = QConv2dReLU(16, 32, 3, 1, padding=1)
        self.conv4 = QConv2dReLU(32, 32, 3, 1, padding=1)

        self.conv5 = QConv2dReLU(32, 64, 3, 1, padding=1)
        self.conv6 = QConv2dReLU(64, 64, 3, 1, padding=1)

        self.fc = QLinear(1024, 10)

    def forward(self, x):
        # to-be-done-by-student
        x = self.conv1(torch.quantize_per_tensor(x, self.scale, 0, torch.quint8))
        x = self.conv2(x)

        x = nn.quantized.functional.max_pool2d(x, 2)

        x = self.conv3(x)
        x = self.conv4(x)

        x = nn.quantized.functional.max_pool2d(x, 2)

        x = self.conv5(x)
        x = self.conv6(x)

        x = nn.quantized.functional.max_pool2d(x, 2)

        x = x.flatten(1, -1)
        x = self.fc(x)
        x = torch.dequantize(x)
        # to-be-done-by-student

        return x