import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["loss_LS"]

class loss_LS(nn.Module):
    def __init__(self,c,t,alpha=0.01):
        super(loss_LS, self).__init__()
        self.c = c
        self.t = t
        self.alpha=alpha
        self.eps = 0.1  # 标签平滑系数

    def forward(self, pred,label,ori):
        predictions = F.softmax(pred, dim=1)

        # 标签平滑
        n_classes = predictions.size(1)
        one_hot = torch.zeros_like(predictions).scatter_(1, label.unsqueeze(1), 1)

        # 标签平滑：目标类别的概率变为 (1 - epsilon)，非目标类别的概率变为 epsilon / (C-1)
        smooth_target = one_hot * (1 - self.eps) + (1 - one_hot) * (self.eps / (n_classes - 1))

        # 计算标签平滑后的交叉熵损失
        lprobs = F.log_softmax(pred, dim=1)
        loss_ce = -torch.sum(smooth_target * lprobs, dim=1)  # 平滑标签下的交叉熵损失
        loss_ce = loss_ce.mean()


        loss=  loss_ce
        Y_pred = torch.argmax(pred, 1)

        #print(final,loss_ce)
        return loss,Y_pred

    def __call__(self, pred, label,ori=None):
        return self.forward(pred, label,ori)