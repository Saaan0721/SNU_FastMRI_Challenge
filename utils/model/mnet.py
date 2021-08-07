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

        
        self.first_block = ConvBlock(in_chans, 2)
        self.ll1 = MaxPool()
        self.down1 = Down(2, 4)
        self.up1 = Up(4, 2)
        self.last_block = nn.Conv2d(2, out_chans, kernel_size=1)

        self.concat = Concat()
        


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

        d1 = self.first_block(input)
        ll2 = self.ll1(input)
        d2 = self.down1(d1, ll2)
        u1 = self.up1(d2, d1)
        u1 = self.concat(u1, d1)

        
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
        x1 = self.conv(x)
        x2 = torch.cat([x1, x], dim=1)
        return self.conv(x2)


class Down(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.down = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_chans, out_chans)

    def forward(self, x, concat_input):
        x = self.down(x)
        concat_output = torch.cat([concat_input, x], dim=1)
        return self.conv(concat_output)


class MaxPool(nn.Module):

    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.maxpool(x)


class Up(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.up = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_chans, out_chans)

    def forward(self, x, concat_input):
        x = self.up(x)
        concat_output = torch.cat([concat_input, x], dim=1)
        return self.conv(concat_output)


class UpSample(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.up = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class Concat(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, concat_input):
        concat_output = torch.cat([concat_input, x], dim=1)
        return self.conv(concat_output)
