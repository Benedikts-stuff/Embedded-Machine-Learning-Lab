import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyYoloV2Fused(nn.Module):

    def __init__(self, num_classes=20):
        super().__init__()

        anchors = ((1.08, 1.19),
                   (3.42, 4.41),
                   (6.63, 11.38),
                   (9.42, 5.11),
                   (16.62, 10.52),)

        self.register_buffer("anchors", torch.tensor(anchors))
        self.num_classes = num_classes

        self.pad = nn.ReflectionPad2d((0, 1, 0, 1))

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=True)

        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=True)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=True)

        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1, bias=True)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1, bias=True)

        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1, bias=True)

        self.conv7 = nn.Conv2d(512, 1024, 3, 1, 1, bias=True)

        self.conv8 = nn.Conv2d(1024, 1024, 3, 1, 1, bias=True)

        self.conv9 = nn.Conv2d(1024, len(anchors) * (5 + num_classes), 1, 1, 0)

    def forward(self, x, yolo=True):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv4(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv5(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv6(x)
        x = self.pad(x)
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv7(x)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv8(x)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv9(x)
        if yolo:
            nB, _, nH, nW = x.shape

            x = x.view(nB, self.anchors.shape[0], -1, nH, nW).permute(0, 1, 3, 4, 2)

            anchors = self.anchors.to(dtype=x.dtype, device=x.device)
            range_y, range_x, = torch.meshgrid(
                torch.arange(nH, dtype=x.dtype, device=x.device),
                torch.arange(nW, dtype=x.dtype, device=x.device)
            )
            anchor_x, anchor_y = anchors[:, 0], anchors[:, 1]

            x = torch.cat([
                (x[:, :, :, :, 0:1].sigmoid() + range_x[None, None, :, :, None]) / nW,  # x center
                (x[:, :, :, :, 1:2].sigmoid() + range_y[None, None, :, :, None]) / nH,  # y center
                (x[:, :, :, :, 2:3].exp() * anchor_x[None, :, None, None, None]) / nW,  # Width
                (x[:, :, :, :, 3:4].exp() * anchor_y[None, :, None, None, None]) / nH,  # Height
                x[:, :, :, :, 4:5].sigmoid(),  # confidence
                x[:, :, :, :, 5:].softmax(-1), ], -1)

        return x
