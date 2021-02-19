import torch
import torchvision

def get_dataloader(args):
    train_ds, train_dl = get_dataset(args)