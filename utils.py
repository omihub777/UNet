import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import glob
from dataset import RealCropDataset
from model.unet import UNet
from criterions import DiceBCELoss

def get_criterion(args):
    print(f"Loss: {args.loss}")
    if args.loss=="mse":
        criterion = nn.MSELoss()
    elif args.loss=="bce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss=="dice":
        criterion = DiceBCELoss(weight=args.dice_weight)
    else:
        raise ValueError(f"{args.loss}?")
    return criterion

def get_dataloader(args):
    train_ds, test_ds = get_dataset(args)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_dl, test_dl

def get_dataset(args):
    train_path = glob.glob("data/train/*.jpg")
    test_path = glob.glob("data/test/*.jpg")
    train_ds = RealCropDataset(train_path, train=True, size=args.size)
    test_ds = RealCropDataset(test_path, train=False, size=args.size)
    return train_ds, test_ds

def get_model(args):
    print(f"Model: {args.model_name}")
    if args.model_name=="unet":
        model = UNet(args.in_c, args.out_c)
    else:
        raise ValueError(f"{args.model_name}?")
    return model

def get_optimizer(args, model):
    print(f"Optimizer: {args.optimizer}")
    if args.optimizer=="adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
    else:
        raise ValueError(f"{args.optimizer}?")
    return optimizer

def get_experiment_name(args):
    experiment_name = f"{args.model_name}_{args.loss}"

    print(f"Experiment: {experiment_name}")
    return experiment_name

if __name__=="__main__":
    import argparse
    args = argparse.Namespace()
    args.size = 224
    args.batch_size = 16
    args.eval_batch_size = 16
    train_dl, test_dl = get_dataloader(args)
    image, target = next(iter(train_dl))
    print(image.shape, target.shape)
