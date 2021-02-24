import torch
import torch.nn as nn
import torch.nn.functional as F



# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
class DiceBCELoss(nn.Module):
    """Binary Cross Entropy Loss with Dice Loss.
        :param weight: weight for dice_loss
    """
    def __init__(self, weight: float = 1.):
        super(DiceBCELoss, self).__init__()
        self.weight = weight

    def forward(self, out, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # out = F.sigmoid(out)
        
        #flatten label and prediction tensors
        out = out.view(-1)
        targets = targets.view(-1)
        
        intersection = (out * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(out.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy_with_logits(out, targets, reduction='mean')
        Dice_BCE = BCE + self.weight*dice_loss
        
        return Dice_BCE