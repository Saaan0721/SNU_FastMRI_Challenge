import torch
import numpy as np
import random


def seed_fix(int):
    #PyTorch
    torch.manual_seed(int)
    torch.cuda.manual_seed(int)
    torch.cuda.manual_seed_all(int) #multi-GPU
    #CuDNN
    torch.backends.cudnn.deterministics = True
    torch.backends.cudnn.benchmark = False
    #Numpy
    np.random.seed(int)
    #Random
    random.seed(int)
