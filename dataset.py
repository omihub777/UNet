import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

class RealCropDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, train=True, size=224):
        super(RealCropDataset, self).__init__()
        self.img_path = img_path
        self.train = train
        if self.train:
            self.target_path = [img_p.replace("train","target").replace("jpg","png") for img_p in self.img_path]
        self.size = size

    def _transform(self, image, target):
        image = transforms.functional.resize(image, size=(self.size, self.size))
        target = transforms.functional.resize(target, size=(self.size, self.size))

        image = transforms.functional.pad(image, padding=self.size//8)
        target = transforms.functional.pad(target, padding=self.size//8)

        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(self.size, self.size)
        )
        image = transforms.functional.crop(image, i,j,h,w)
        target = transforms.functional.crop(target, i, j, h, w)

        if torch.rand(1) > 0.5:
            image = transforms.functional.hflip(image)
            target = transforms.functional.hflip(target)

        if torch.rand(1) > 0.5:
            image = transforms.functional.vflip(image)
            target = transforms.functional.vflip(target)

        image = transforms.functional.to_tensor(image)
        target = transforms.functional.to_tensor(target)

        image = transforms.functional.normalize(image, mean=[.5,.5,.5], std=[.5,.5,.5])
        # target = transforms.functional.normalize(target, mean=[.5], std=[.5])

        return image, target

    def _test_transform(self, image):
        image = transforms.functional.resize(image, size=(self.size, self.size))
        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(image, mean=[.5,.5,.5], std=[.5,.5,.5])
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