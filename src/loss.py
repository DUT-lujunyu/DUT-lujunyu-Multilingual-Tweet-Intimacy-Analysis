import torch.nn.functional as F
import torch, math
import torch.nn as nn

class Intimacy_Loss(torch.nn.Module):
    def __init__(self):
        super(Intimacy_Loss, self).__init__()

        self.loss_function = torch.nn.L1Loss()
        # torch.nn.MSELoss torch.nn.SmoothL1Loss torch.nn.HuberLoss

    def forward(self, pred, label):
        loss = self.loss_function(pred, label)
        return loss