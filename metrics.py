import torch

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

if __name__ == "__main__":
    b,c,h,w = 4, 3, 16, 16
    # out = torch.randn(b, c, h, w)
    out = torch.zeros(b,c,h,w)
    target = torch.randint(0, 2, size=(b, c, h, w)).float()
    print(dice_fn(out, target))