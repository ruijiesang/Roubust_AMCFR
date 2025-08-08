import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["loss_FR_LS"]

class loss_FR_LS(nn.Module):
    def __init__(self,c,t,alpha=0.01):
        super(loss_FR_LS, self).__init__()
        self.c = c
        self.t = t
        self.alpha=alpha
        self.eps = 0.1  # 标签平滑系数

    def forward(self, pred,label,ori):
        predictions = F.softmax(pred, dim=1)
        batch_size = predictions.size(0)

        # smooth_label
        n_classes = predictions.size(1)
        one_hot = torch.zeros_like(predictions).scatter_(1, label.unsqueeze(1), 1)

        # smooth_label：the probability of the target category has changed to (1 - epsilon)
        smooth_target = one_hot * (1 - self.eps) + (1 - one_hot) * (self.eps / (n_classes - 1))

        # Calculate the cross-entropy loss after smoothing the labels
        lprobs = F.log_softmax(pred, dim=1)
        loss_ce = -torch.sum(smooth_target * lprobs, dim=1)
        loss_ce = loss_ce.mean()

        epsilon = 0.01
        entropy = predictions * torch.log(predictions + epsilon)
        entropy = torch.sum(entropy, dim=1)

        entropy = entropy / torch.log(torch.tensor(self.c, dtype=predictions.dtype))
        entropy = torch.sum(entropy)
        entropy = -entropy / batch_size
        topk_values, _ = torch.topk(predictions, k=self.t, dim=1)
        T_loss = torch.sum(topk_values ** 2, dim=1)

        #right_factor = entropy / (torch.sqrt(2 * torch.tensor(torch.pi))*T_loss + epsilon)
        right_factor = entropy / (torch.sqrt(2 * torch.tensor(torch.pi)) * ((T_loss - int(self.t) * ((torch.mean(topk_values, dim=1)) ** 2) + 0.01) / (
                    torch.sum(topk_values) ** 2 - int(self.t) * ((torch.mean(topk_values, dim=1)) ** 2))) + epsilon)
        left_factor = torch.exp((torch.log((T_loss-int(self.t)*((torch.mean(topk_values,dim=1))**2))/(torch.sum(topk_values)**2-int(self.t)*((torch.mean(topk_values,dim=1))**2))+0.01)**2*entropy**2)/2+epsilon)
        final_loss = right_factor * left_factor

        final = torch.mean(final_loss)

        #print(a)
        loss= self.alpha *final + loss_ce
        Y_pred = torch.argmax(pred, 1)

        #print(final,loss_ce)
        return loss,Y_pred

    def __call__(self, pred, label,ori=None):
        return self.forward(pred, label,ori)