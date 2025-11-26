import torch
import torch.nn as nn
import torch.nn.functional as F


class CifarNet(torch.nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16,16, 3, 1, padding=1)

        self.conv3 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, padding=1)

        self.conv5 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, padding=1)

        self.bn1 = nn.BatchNorm2d(16, track_running_stats=False)
        self.bn2 = nn.BatchNorm2d(16, track_running_stats=False)
        self.bn3 = nn.BatchNorm2d(32, track_running_stats=False)
        self.bn4 = nn.BatchNorm2d(32, track_running_stats=False)
        self.bn5 = nn.BatchNorm2d(64, track_running_stats=False)
        self.bn6 = nn.BatchNorm2d(64, track_running_stats=False)

        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2, stride=2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2, stride=2)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2, stride=2)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
