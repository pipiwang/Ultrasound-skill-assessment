import torch.nn as nn
import surface_distance as sd

class DiceLoss(nn.Module):
    def __init__(self, weight = None, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.weight = weight

    def forward(self, pred, target, ):
        batch_size = pred.size(0)
        num_classes = pred.size(1)

        pred = pred.view(batch_size, num_classes, -1)
        target = target.view(batch_size, num_classes, -1)

        intersection = (pred * target).sum(-1)
        denominator = pred.sum(-1) + target.sum(-1)

        if self.weight is None:
            loss = 1. - (2. * intersection) / (denominator + self.eps)
        else:
            loss = 1. - (2. * intersection) / (denominator + self.eps) * self.weight

        return loss.mean()

class WeightedDiceLoss(nn.Module):
    def __init__(self, weight=None, eps=1e-6):
        super(WeightedDiceLoss, self).__init__()
        self.eps = eps
        self.weight = weight

    def forward(self, pred, target, score):
        batch_size = pred.size(0)
        num_classes = pred.size(1)

        pred = pred.view(batch_size, num_classes, -1)
        target = target.view(batch_size, num_classes, -1)

        intersection = (pred * target).sum(-1)
        denominator = pred.sum(-1) + target.sum(-1)

        if self.weight is None:
            loss = 1. - (2. * intersection) / (denominator + self.eps)
        else:
            loss = 1. - (2. * intersection) / (denominator + self.eps) * self.weight

        loss = loss.mean(dim=1) # (bs,)

        return (loss * score).mean()

def binary_dice(y_pred, y_true, thre=0.5, eps=1e-6):
    y_true = y_true >= thre
    y_pred = y_pred >= thre

    intersection = (y_true * y_pred).sum()
    dice = (2. * intersection) / (y_true.sum() + y_pred.sum() +eps)

    return dice

def mult_binary_dice(y_pred, y_true, thre=0.5, eps = 1e-6):
    assert len(y_pred.shape)==4, 'This binary dice only takes input tensor in shape (b,c,x,x,)'

    batch_size = y_pred.size(0)
    num_classes = y_pred.size(1)

    y_true = y_true >= thre
    y_pred = y_pred >= thre

    y_pred = y_pred.view(batch_size, num_classes, -1)
    y_true = y_true.view(batch_size, num_classes, -1)

    intersection = (y_true * y_pred).sum(-1)
    dice = (2. * intersection) / (y_true.sum(-1) + y_pred.sum(-1) + eps)

    return dice

def calc_dice(y_pred, y_true, thre=0.5, eps=1e-6):
    intersection = (y_true * y_pred).sum()
    dice = (2. * intersection) / (y_true.sum() + y_pred.sum() + eps)
    return dice

def calc_assd(y_pred, y_true, thre=0.5):
    assert len(y_pred.shape) == 2, 'This metric only takes input tensor in shape (x,x,)'

    y_true = y_true >= thre
    y_pred = y_pred >= thre

    surfance_distance = sd.compute_surface_distances(y_true, 
                                                     y_pred, 
                                                     spacing_mm=(1,1))
    assd = sd.compute_average_surface_distance(surfance_distance)
    
    return sum(assd)/len(assd)
                