import torch
import numpy as np

@torch.no_grad()
def dice_fn(out, target):
    """Calculate Dice coefficient.

    ```
    Dice = 2*|A&B|/(|A|+|B|)
    ```
    """
    b,c,h,w = target.size()
    hard_out = out.clamp(0,1).round().view(b,-1)
    target = target.view(b, -1)
    numer = 2.*(hard_out*target).sum(dim=-1)
    denom = torch.sum(hard_out,dim=-1)+torch.sum(target,dim=-1)
    dice = torch.mean(numer/denom)
    # import IPython; IPython.embed(); exit(1)
    return dice

def calculate_stat_helper(predict, label):
    test1 = predict.numpy()
    test2 = label.numpy()
    # import IPython; IPython.embed(); exit(1)
    tp = np.sum(np.logical_and(test1 >= 0.5, test2 >= 0.5))
    tn = np.sum(np.logical_and(test1 < 0.5, test2 < 0.5))
    fp = np.sum(np.logical_and(test1 >= 0.5, test2 < 0.5))
    fn = np.sum(np.logical_and(test1 < 0.5, test2 >= 0.5))
    iou = tp/(tp+fp+fn)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    fvalue = 2/((1/precision)+(1/recall))
    return precision, recall, fvalue, iou

if __name__ == "__main__":
    b,c,h,w = 4, 3, 16, 16
    out = torch.randn(b, c, h, w)
    # out = torch.zeros(b,c,h,w)
    target = torch.randint(0, 2, size=(b, c, h, w)).float()
    print(dice_fn(out, target))
    print(calculate_stat_helper(out.numpy(), target.numpy()))