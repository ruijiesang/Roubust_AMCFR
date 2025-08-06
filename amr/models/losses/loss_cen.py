import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["loss_cen"]


class loss_cen(nn.Module):
    def __init__(self,c,t,alpha=0.00001):
        super(loss_cen, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.alpha=0.001

    def forward(self, pred, label, ori):
        predictions = F.softmax(pred, dim=1)
        epsilon = 0.00000001
        entropy = predictions * torch.log(predictions + epsilon)
        entropy=-torch.sum(entropy)
        loss_ce = self.ce(pred, label)
        loss=self.alpha*entropy+loss_ce
        Y_pred = torch.argmax(pred, 1)


        return loss, Y_pred

    def __call__(self, pred, label, ori=None):
        return self.forward(pred, label, ori)