import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", default="unet")
parser.add_argument("--model-path", required=True)
args = parser.parse_args()
