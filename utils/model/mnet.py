import torch
from torch import nn
from torch.nn import functional as F


class Mnet(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        '''
        self.first_block = ConvBlock(in_chans, 2)
        self.down1 = Down(2, 4)
        self.up1 = Up(4, 2)
        self.last_block = nn.Conv2d(2, out_chans, kernel_size=1)
        '''

        self.first = nn.Conv2d(in_chans, 16, kernel_size=1)
        self.first_block = FirstBlock(16, 32)
        self.down1 = Down(32, 64, 16)
        self.down2 = Down(64, 128, 16)
        self.down3 = Down(128, 256, 16)
        self.down4 = Down(256, 512, 16)
        self.middle_block = ConvBlock(512, 256)
        self.up4 = Up(256, 128, 256)
        self.up3 = Up(128, 64, 128)
        self.up2 = Up(64, 32, 64)
        self.up1 = Up(32, 16, 32)
        self.last_block = nn.Conv2d(256+128+64+32+16, out_chans, kernel_size=1)

        self.ll = MaxPool()
        self.rl4 = UpSample(256)
        self.rl3 = UpSample(256+128)
        self.rl2 = UpSample(256+128+64)
        self.rl1 = UpSample(256+128+64+32)
        


    def norm(self, x):
        b, h, w = x.shape
        x = x.view(b, h * w)
        mean = x.mean(dim=1).view(b, 1, 1)
        std = x.std(dim=1).view(b, 1, 1)
        x = x.view(b, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, input):
        input, mean, std = self.norm(input)
        input = input.unsqueeze(1)

        '''
        d1 = self.first_block(input)
        m0 = self.down1(d1)
        u1 = self.up1(m0, d1)
        '''

        d0 = self.first(input)

        d1 = self.first_block(d0)
        ll2 = self.ll(d0)
        d2 = self.down1(d1, ll2)
        ll3 = self.ll(ll2)
        d3 = self.down2(d2, ll3)
        ll4 = self.ll(ll3)
        d4 = self.down3(d3, ll4)
        ll5 = self.ll(ll4)
        d5 = self.down4(d4, ll5)

        d5 = self.middle_block(d5)

        u4 = self.up4(d5, d4)
        rl4 = self.rl4(d5, u4)
        u3 = self.up3(u4, d3)
        rl3 = self.rl3(rl4, u3)
        u2 = self.up2(u3, d2)
        rl2 = self.rl2(rl3, u2)
        u1 = self.up1(u2, d1)
        u1 = self.rl1(rl2, u1)

        
        output = self.last_block(u1)
        output = output.squeeze(1)
        output = self.unnorm(output, mean, std)

        return output


class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class FirstBlock(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.conv1 = ConvBlock(in_chans, in_chans)
        self.conv2 = ConvBlock(out_chans, out_chans)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.cat([x1, x], dim=1)
        return self.conv2(x2)


class Down(nn.Module):

    def __init__(self, in_chans, out_chans, concat_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.concat_chans = concat_chans
        self.down = nn.MaxPool2d(2)
        self.conv1 = ConvBlock(in_chans+concat_chans, in_chans)
        self.conv2 = ConvBlock(out_chans, out_chans)

    def forward(self, x, concat_input):
        x1 = self.down(x)
        x2 = torch.cat([concat_input, x1], dim=1)
        x2 = self.conv1(x2)
        x3 = torch.cat([x2, x1], dim=1)
        return self.conv2(x3)


class MaxPool(nn.Module):

    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.maxpool(x)


class Up(nn.Module):

    def __init__(self, in_chans, out_chans, concat_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.concat_chans = concat_chans
        self.up = nn.ConvTranspose2d(in_chans, in_chans, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_chans+concat_chans, in_chans)
        self.conv2 = ConvBlock(in_chans+in_chans, out_chans)

    def forward(self, x, concat_input):
        x1 = self.up(x)
        x2 = torch.cat([concat_input, x1], dim=1)
        x2 = self.conv1(x2)
        x3 = torch.cat([x2, x1], dim=1)
        return self.conv2(x3)


class UpSample(nn.Module):

    def __init__(self, in_chans):
        super().__init__()
        self.in_chans = in_chans
        self.up = nn.ConvTranspose2d(in_chans, in_chans, kernel_size=2, stride=2)

    def forward(self, x, concat_input):
        up = self.up(x)
        return torch.cat([concat_input, up], dim=1)

