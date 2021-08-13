from torch import nn
from utils.model.unet import Unet
from utils.model.mnet import Mnet


class MUnet(nn.Module):
    def __init__(self, in_chans, out_chans):
          super().__init__()
          self.in_chans = in_chans
          self.out_chans = out_chans

          self.layer = nn.Sequential(
              Mnet(in_chans, 1),
              Unet(1, out_chans)
          )


    def forward(self, input):
        return self.layer(input)
