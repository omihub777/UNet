import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

class RealCropDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, train=True, size=224):
        super(RealCropDataset, self).__init__()
        self.img_path = img_path
        self.train = train
        if self.train:
            self.target_path = [img_p.replace("train","target").replace("jpg","png") for img_p in self.img_path]
        self.size = size

    def _transform(self, image, target):
        image = TF.resize(image, size=(self.size, self.size))
        target = TF.resize(target, size=(self.size, self.size))

        #Pad
        image = TF.pad(image, padding=self.size//8)
        target = TF.pad(target, padding=self.size//8)

        #Crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(self.size, self.size)
        )
        image = TF.crop(image, i,j,h,w)
        target = TF.crop(target, i, j, h, w)

        #HFlip
        if np.random.random(1) > 0.5:
            image = TF.hflip(image)
            target = TF.hflip(target)

        #VFlip
        if np.random.random(1) > 0.5:
            image = TF.vflip(image)
            target = TF.vflip(target)

        # Rotation
        angle = np.random.randint(low=-45, high=45)
        image = TF.rotate(image, angle)
        target = TF.rotate(target, angle)
        
        # ====Only for image!====
        # ColorJitter(Brightness/Contrast/Saturation/Hue)
        image = transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.1)(image)

        # GrayScale
        if np.random.random(1) > 0.3:
            image = TF.to_grayscale(image, num_output_channels=3)

        # Gaussian Blur(for motion noise or some ill-setting)
        # image = transforms.GaussianBlur(kernel_size=3)(image)
    
        # ====Up to here!====


        # Convert to torch.Tensor
        image = TF.to_tensor(image)
        target = TF.to_tensor(target)

        # Normalize image
        image = TF.normalize(image, mean=[.5,.5,.5], std=[.5,.5,.5])

        return image, target

    def _test_transform(self, image):
        image = TF.resize(image, size=(self.size, self.size))
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[.5,.5,.5], std=[.5,.5,.5])
        return image

    def __getitem__(self, idx):
        img_p = self.img_path[idx]
        img = Image.open(img_p).convert("RGB")
        if self.train:
            target_p = self.target_path[idx]
            target = Image.open(target_p).convert("L")
            img, target = self._transform(img, target)
            return img, target
        else:
            img = self._test_transform(img)
            return img

    def __len__(self):
        return len(self.img_path)