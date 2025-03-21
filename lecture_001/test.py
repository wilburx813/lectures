import torch

def square(a):
    return torch.square(a)

opt_square = torch.compile(square)

opt_square(torch.randn(10000,10000).cuda())