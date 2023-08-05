import glob
import argparse
from torch.utils.data import DataLoader

from losses import *
from dataset_code import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--crop-size', default=88, type=int, help='training images crop size')
    parser.add_argument('--crop-required', default=False, type=bool, help='cropping is required or not')
    parser.add_argument('--upscale-factor', default=4, type=int, choices=[2, 4, 8],
                        help='super resolution upscale factor')
    parser.add_argument('--num-epochs', default=100, type=int, help='train epoch number')
    parser.add_argument('--pretrained', default=False, type=bool, help='should use pretrained model for training')
    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    CROP_REQUIRED = opt.crop_required
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    PRETRAINED = opt.pretrained

    image_file_paths = glob.glob(pathname=dataset_path + "/*.*")[:1000]
    train_image_count = int(len(image_file_paths) * 0.8)
    train_file_paths = image_file_paths[:train_image_count]
    val_file_paths = image_file_paths[train_image_count:]
    train_data_loader = DataLoader(
        dataset=TrainValDataset(train_file_paths, CROP_SIZE, UPSCALE_FACTOR, CROP_REQUIRED),
        batch_size=train_batch_size,
        shuffle=False,
    )
    val_data_loader = DataLoader(
        dataset=TrainValDataset(val_file_paths, CROP_SIZE, UPSCALE_FACTOR, CROP_REQUIRED),
        batch_size=val_batch_size,
        shuffle=False,
    )

    if not PRETRAINED:
        generator = GeneratorResNet(scale_factor=UPSCALE_FACTOR)
        discriminator = Discriminator()
    else:
        generator = torch.load('./saved_models/generator.pth')
        discriminator = torch.load('./saved_models/discriminator.pth')

    print(f'generator parameters: {sum(param.numel() for param in generator.parameters())}')
    print(f'discriminator parameters: {sum(param.numel() for param in discriminator.parameters())}')

    criterion_generator = GeneratorLoss()

    if device is not None:
        generator = generator.to(device)
        discriminator = discriminator.to(device)
        criterion_content = criterion_generator.to(device)

    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=learning_rate, betas=(b1, b2)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=learning_rate, betas=(b1, b2)
    )

    train_val(
        NUM_EPOCHS,
        train_data_loader,
        val_data_loader,
        generator,
        discriminator,
        optimizer_G,
        optimizer_D,
        criterion_generator,
        device,
        UPSCALE_FACTOR,
    )
