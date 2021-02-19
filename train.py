import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=1e-3)
args = parser.parse_args()


train_dl, test_dl = get_dataloader()