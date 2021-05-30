import math
import torch


def gelu(x):
    # gaussian error linear units
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return cdf*x