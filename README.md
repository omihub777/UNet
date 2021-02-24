# UNet

Re-implementation of U-Net[[Ronneberger, O(MICCAI15)]](https://arxiv.org/abs/1505.04597) in PyTorch.
In `unet.py`, while `UNetVannila` follows the paper's architecture(i.e. convolution layers don't have padding.), `UNet` has convolutions which use padding.
In this repo, we use `UNet` mainly.


## TODO
* Double Check duplciation between train and test data.
* Add More DAs
    * Rotation(45)
    * Color Jitter(Brightness/Contreast/Saturation/Hue)
    * Gaussian Blur?
* From ConvBlock to ResBlock in bottleneck.
* Dilated Conv?
* test.py

## Done
* get_dataset
* get_dataloader
* get_model
* get_optimizer
* get_scheduler
* Trainer
* MSELoss
* BCE+Dice Loss(from [kaggle](https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/40199))
* Lessen the number of parameters([3rd place solution](https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/40199) uses 8M UNet)
