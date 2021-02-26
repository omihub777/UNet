# UNet

Re-implementation of U-Net[[Ronneberger, O(MICCAI15)]](https://arxiv.org/abs/1505.04597) in PyTorch.
In `unet.py`, while `UNetVannila` follows the paper's architecture(i.e. convolution layers don't have padding.), `UNet` has convolutions which use padding.
In this repo, we use `UNet` mainly.


## TODO
* Tackle with "Shaded White removing problem"
    * Can't identify shaded white parts of items.
    * Higher Brightness degree?
* Quantitative Evaluation.(Dice)
* Dilated Conv?
* test.py
* Add Random Grayscale to catch the shape of the objects

## Done
* get_dataset
* get_dataloader
* get_model
* get_optimizer
* get_scheduler
* Trainer
* MSELoss
* BCE+Dice Loss(from [kaggle](https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/40199))
    * Messed up with Loss values.
    * BCE-only works well. So, dice loss degrades the performance.
* Lessen the number of parameters([3rd place solution](https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/40199) uses 8M UNet)
    * Works well.(not better.)
* Double Check duplciation between train and test data.
    * Using diffrent ids b/w train and test data.
* BCELoss
    * In the literature, every paper uses bce rather than mse. We stick to this.
* Add More DAs
    * Rotation(45)
    * Color Jitter(Brightness/Contreast/Saturation/Hue)
    * Gaussian Blur?
* From ConvBlock to ResBlock (in bottleneck.)
    * Ref:[DeepResUNet](https://arxiv.org/abs/1711.10684)
        * Use pre-act resblock for all blocks.(not only bottleneck)
