import torch
from torch import nn
from torchvision.models.vgg import vgg16


class PDLoss(nn.Module):
    def __init__(self):
        super(PDLoss, self).__init__()
        self.mse_loss = nn.MSELoss(size_average=False)


    def forward(self, res, difference):
        n = len(res)
        difference = difference / n
        image_loss = 0
        for k in range(n):
            # Image Loss
            image_loss += self.mse_loss(res[k], difference)/difference.shape[0]
        return image_loss

#PDLoss2 is a variant version of PDLoss
class PDLoss2(nn.Module):
    def __init__(self):
        super(PDLoss2, self).__init__()
        self.mse_loss = nn.MSELoss(size_average=False)


    def forward(self, res, difference):
        n = len(res)
        difference = difference / n
        image_loss = 0
        for k in range(n):
            # Image Loss
            image_loss += self.mse_loss(sum(res[0:k+1]), (k + 1) * difference)/difference.shape[0]
        return image_loss

class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        self.L1loss = nn.L1Loss()

    def forward(self, x, y):
        image_loss = self.L1loss(x, y)/y.shape[0]
        return image_loss


class L1Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(L1Loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return F.l1_loss(input, target, reduction=self.reduction)

if __name__ == "__main__":
    g_loss = PDLoss()
    print(g_loss)
