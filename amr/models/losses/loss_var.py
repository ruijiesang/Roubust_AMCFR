import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["loss_var"]


class loss_var(nn.Module):
    def __init__(self):
        super(loss_var, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.alpha=1
        self.c=11

    def forward(self, pred, label, ori):
        predictions = F.softmax(pred, dim=1)
        var=torch.pow(predictions-(1/self.c),2)
        var=torch.sum(var)
        loss_ce = -self.ce(pred, label)
        loss=self.alpha*var+loss_ce
        Y_pred = torch.argmax(pred, 1)


        return loss, Y_pred

    def __call__(self, pred, label, ori=None):
        return self.forward(pred, label, ori)