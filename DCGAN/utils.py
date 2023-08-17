import random
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import RandomSampler, Subset

from consts import *
import sys


def get_celeb(params):
    data_transforms = transforms.Compose(
        [
            transforms.Resize(params["img_size"]),
            transforms.CenterCrop(params["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.ImageFolder(root=data_root_folder, transform=data_transforms)
    print(len(dataset))
    sample_ds = Subset(dataset, np.random.randint(0, len(dataset), size=(5000,)))
    sample_sampler = RandomSampler(sample_ds)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params["batch_size"],
        sampler=sample_sampler,
        drop_last=True,
    )
    return dataloader


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    loader = get_celeb({"img_size": 64, "batch_size": 2})
    print(f"total images found: {len(loader.dataset)}")
    batch = next(iter(loader))
    print(batch, batch[0].shape)
