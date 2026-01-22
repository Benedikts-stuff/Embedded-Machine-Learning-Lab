import torch
import torch.nn as nn
import torch.nn.functional as F
from Training.quantization import QConv2dLeakyReLU



class QTinyYoloV2(nn.Module):
    def __init__(self, channels, num_classes=1):
        super().__init__()
        self.register_buffer("in_scale", torch.tensor(0.1))
        self.register_buffer("in_zp", torch.tensor(0))

        self.pad = nn.ReflectionPad2d((0, 1, 0, 1))

        self.register_buffer('anchors', torch.tensor([
            [1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]
        ]))
        
        #channels from densify_yolo_state_dict
        self.qconv1 = QConv2dLeakyReLU(3,           channels[0], 3, 1, 1,)
        self.qconv2 = QConv2dLeakyReLU(channels[0], channels[1], 3, 1, 1)
        self.qconv3 = QConv2dLeakyReLU(channels[1], channels[2], 3, 1, 1)
        self.qconv4 = QConv2dLeakyReLU(channels[2], channels[3], 3, 1, 1)
        self.qconv5 = QConv2dLeakyReLU(channels[3], channels[4], 3, 1, 1)
        self.qconv6 = QConv2dLeakyReLU(channels[4], channels[5], 3, 1, 1)
        self.qconv7 = QConv2dLeakyReLU(channels[5], channels[6], 3, 1, 1)
        self.qconv8 = QConv2dLeakyReLU(channels[6], channels[7], 3, 1, 1)
        
        self.qconv9 = QConv2dLeakyReLU(channels[7], 5 * (5 + num_classes), 1, 1, 0, use_relu=False)


    def repack_all(self):
        for i in range(1, 10):
            getattr(self, f"qconv{i}").repack_weights()

    def forward(self, x, yolo=True):
        x = torch.quantize_per_tensor(x, self.in_scale, self.in_zp, torch.quint8)

        x = nn.quantized.functional.max_pool2d(self.qconv1(x), 2, 2)
        x = nn.quantized.functional.max_pool2d(self.qconv2(x), 2, 2)
        x = nn.quantized.functional.max_pool2d(self.qconv3(x), 2, 2)
        x = nn.quantized.functional.max_pool2d(self.qconv4(x), 2, 2)
        x = nn.quantized.functional.max_pool2d(self.qconv5(x), 2, 2)

        x = self.qconv6(x)
        x = torch.nn.functional.pad(x, (0, 1, 0, 1), mode='constant', value=int(self.qconv6.relu_zp))
        x = nn.quantized.functional.max_pool2d(x, 2, 1)

        x = self.qconv7(x)
        x = self.qconv8(x)

        x = self.qconv9(x)

        # INT8 -> Float 
        x = torch.dequantize(x)

        if yolo:
            nB, _, nH, nW = x.shape

            x = x.view(nB, self.anchors.shape[0], -1, nH, nW).permute(0, 1, 3, 4, 2)

            anchors = self.anchors.to(dtype=x.dtype, device=x.device)
            range_y, range_x, = torch.meshgrid(
                torch.arange(nH, dtype=x.dtype, device=x.device),
                torch.arange(nW, dtype=x.dtype, device=x.device), 
                indexing="ij"
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