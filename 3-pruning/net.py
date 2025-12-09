import torch
import torch.nn as nn
import torch.nn.functional as F


class CifarNetTraining(torch.nn.Module):
    def __init__(self):
        super(CifarNetTraining, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, padding=1)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, padding=1)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(1024, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)

        x = F.max_pool2d(x, 2, stride=2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x)

        x = F.max_pool2d(x, 2, stride=2)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.leaky_relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.leaky_relu(x)

        x = F.max_pool2d(x, 4, stride=4)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

class CifarNet(torch.nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16,32, 3, 1, padding=1)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, padding=1)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, padding=1)

        self.fc = nn.Linear(1024, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)

        x = F.max_pool2d(x, 2, stride=2)

        x = self.conv3(x)
        x = F.leaky_relu(x)

        x = self.conv4(x)
        x = F.leaky_relu(x)

        x = F.max_pool2d(x, 2, stride=2)

        x = self.conv5(x)
        x = F.leaky_relu(x)

        x = self.conv6(x)
        x = F.leaky_relu(x)

        x = F.max_pool2d(x, 4, stride=4)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

class PrunedCifarNet(CifarNet):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16,32, 3, 1, padding=1)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, padding=1)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, padding=1)

        self.fc = nn.Linear(1024, 10)

        self._register_load_state_dict_pre_hook(self._sd_hook)

    def _sd_hook(self, state_dict, prefix, *_):
        for key in state_dict:
            if 'conv' in key and 'weight' in key:
                n = int(key.split('conv')[1].split('.weight')[0])

            dim_in = state_dict[f'conv{n}.weight'].shape[1]
            dim_out = state_dict[f'conv{n}.weight'].shape[0]

            conv = nn.Conv2d(dim_in, dim_out, 3, 1, padding=1)
            if n == 1: self.conv1 = conv
            elif n == 2: self.conv2 = conv
            elif n == 3: self.conv3 = conv
            elif n == 4: self.conv4 = conv
            elif n == 5: self.conv5 = conv
            elif n == 6: self.conv6 = conv
            self.fc = nn.Linear(state_dict['fc.weight'].shape[1], 10)
        pass

if __name__ == '__main__':

    net = CifarNet()

    net(torch.zeros(1, 3, 32, 32))