import torch
import torch.nn as nn
import time

class Bottleneck(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1,t=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channel,in_channel*t,kernel_size=1,bias=False),
            nn.BatchNorm2d(in_channel*t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channel*t,in_channel*t,kernel_size=3,stride=stride,padding=1,groups=in_channel*t,bias=False),
            nn.BatchNorm2d(in_channel*t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channel*t,out_channel,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channel),

        )
    def forward(self,x):
        if self.stride ==1 and self.in_channel == self.out_channel:
            return self.sub_module(x)+x
        else:
            return self.sub_module(x)
class UpsampleLayer(torch.nn.Module):

    def __init__(self):
        super(UpsampleLayer, self).__init__()

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
class ConvolutionalSet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSet, self).__init__()

        self.sub_module = nn.Sequential(
            Bottleneck(in_channels, out_channels,stride=1,t=1),
            Bottleneck(out_channels, out_channels,stride=1,t=1),
        )

    def forward(self, x):
        return self.sub_module(x)
class MainNet(nn.Module):

    def __init__(self,clsnum):
        super(MainNet, self).__init__()

        self.trunk_52 = torch.nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),  # 208
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            Bottleneck(32,16,1,1),  #208
            Bottleneck(16,24,2,6),
            Bottleneck(24,24,1,6),  #104

            Bottleneck(24, 32, 2, 6),
            Bottleneck(32, 32, 1, 6),
            Bottleneck(32, 32, 1, 6),  # 52
        )
        self.trunk_26 = torch.nn.Sequential(
            Bottleneck(32, 64, 2, 6),
            Bottleneck(64, 64, 1, 6),
            Bottleneck(64, 64, 1, 6),
            Bottleneck(64, 64, 1, 6),
            Bottleneck(64, 96, 1, 6),
            Bottleneck(96, 96, 1, 6),
            Bottleneck(96, 96, 1, 6),  # 26

        )
        self.trunk_13 = torch.nn.Sequential(
            Bottleneck(96, 160, 2, 6),
            Bottleneck(160, 160, 1, 6),
            Bottleneck(160, 512, 1, 6),
            nn.Conv2d(512, 1024, 1, 1), # 13
            nn.BatchNorm2d(1024),
            nn.ReLU6(inplace=True)
        )
        self.convset_13 = torch.nn.Sequential(
            ConvolutionalSet(1024, 512)
        )

        self.detetion_13 = torch.nn.Sequential(
            nn.Conv2d(512, 3*(5+clsnum), 3, 1, 1)
        )

        self.up_26 = torch.nn.Sequential(
            Bottleneck(512, 256,1,1),
            UpsampleLayer()
        )

        self.convset_26 = torch.nn.Sequential(
            ConvolutionalSet(352, 160)
        )

        self.detetion_26 = torch.nn.Sequential(
            nn.Conv2d(160, 3*(5+clsnum), 3, 1, 1)
        )
        self.up_52 = torch.nn.Sequential(
            Bottleneck(160, 96,1,1),
            UpsampleLayer()
        )

        self.convset_52 = torch.nn.Sequential(
            ConvolutionalSet(128, 64)
        )

        self.detetion_52 = torch.nn.Sequential(
            nn.Conv2d(64, 3*(5+clsnum), 3, 1, 1)
        )

    def forward(self, x):
        # start_time = time.time()
        h_52 = self.trunk_52(x)
        h_26 = self.trunk_26(h_52)
        h_13 = self.trunk_13(h_26)

        convset_out_13 = self.convset_13(h_13)
        detetion_out_13 = self.detetion_13(convset_out_13)

        up_out_26 = self.up_26(convset_out_13)

        route_out_26 = torch.cat((up_out_26, h_26), dim=1)
        convset_out_26 = self.convset_26(route_out_26)
        detetion_out_26 = self.detetion_26(convset_out_26)

        up_out_52 = self.up_52(convset_out_26)
        route_out_52 = torch.cat((up_out_52, h_52), dim=1)
        convset_out_52 = self.convset_52(route_out_52)
        detetion_out_52 = self.detetion_52(convset_out_52)

        return detetion_out_13, detetion_out_26, detetion_out_52
if __name__ == '__main__':
    trunk = MainNet(3)

    trunk.eval()
    trunk.cuda().half()
    x = torch.cuda.HalfTensor(1, 3, 416, 416)

    y_13, y_26, y_52 = trunk(x)
    print(y_13.shape)
    print(y_26.shape)
    print(y_52.shape)

    for _ in range(15):
        start_time = time.time()
        trunk(x)
        end_time = time.time()
        print(end_time - start_time)
        print("===================================")