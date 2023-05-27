import torch.nn as nn
import torch.nn.functional as F

class ReconstructionLoss:
    def __init__(self):
        pass

    def __call__(self, x, x_hat):
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss
