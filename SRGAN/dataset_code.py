from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from consts import *


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


class TrainValDataset(Dataset):
    def __init__(self, image_files, crop_size, upscale_factor, crop_required=False):
        super(TrainValDataset, self).__init__()
        self.files = image_files
        self.crop_size = crop_size
        self.crop_required = crop_required
        self.up_scale_factor = upscale_factor
        self.crop_size = calculate_valid_crop_size(self.crop_size, self.up_scale_factor)
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize(
                    size=(self.crop_size // self.up_scale_factor, self.crop_size // self.up_scale_factor),
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(data_mean, data_std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize(
                    size=(self.crop_size, self.crop_size),
                    interpolation=transforms.InterpolationMode.BICUBIC
                ) if not self.crop_required else
                transforms.RandomCrop(
                    size=self.crop_size
                ),
                transforms.ToTensor(),
                transforms.Normalize(data_mean, data_std),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(self.files[index % len(self.files)]).convert(mode="RGB")
        lr_image = self.lr_transform(image)
        hr_image = self.hr_transform(image)
        return {"lr": lr_image, "hr": hr_image}
