# UNet

Re-implementation of U-Net[[Ronneberger, O(MICCAI15)]](https://arxiv.org/abs/1505.04597) in PyTorch.
In `unet.py`, while `UNetVannila` follows the paper's architecture(i.e. convolution layers don't have padding.), `UNet` has convolutions which use padding.
In this repo, we use `UNet` mainly.


## TODO
* test.py

## Done
* get_dataset
* get_dataloader
* get_model
* get_optimizer
* get_scheduler
* Trainer
* MSELoss