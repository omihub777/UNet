import glob

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from dataset import RealCropDataset
from model.unet import UNet, ResUNet
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
    train_ds, val_ds, test_ds = get_dataset(args)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    return train_dl, val_dl,  test_dl

def get_dataset(args):
    ### Old Path
    # train_path = glob.glob("data/train/*.jpg")
    # val_path = glob.glob("data/val/*.jpg")
    # test_path = glob.glob("data/test/*.jpg")


    ### New Path
    train_path = glob.glob("data/new_train/*/*.jpg")
    val_path = glob.glob("data/new_val/*/*.jpg")
    test_path = glob.glob("data/new_test/*/*.jpg")

    train_ds = RealCropDataset(train_path, split='train', size=args.size)
    val_ds = RealCropDataset(val_path, split='val', size=args.size)
    test_ds = RealCropDataset(test_path, split='test', size=args.size)
    return train_ds, val_ds, test_ds
        # image = TF.resize(image, size=(self.size, self.size))
        # target = TF.resize(target, size=(self.size, self.size))

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
    args.num_workers=1
    train_dl,val_dl, test_dl = get_dataloader(args)
    # image, target = next(iter(train_dl))
    image, target = next(iter(val_dl))
    print(image.shape, target.shape)
