import math
from torch import nn
from torchvision import models

from utils import *


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(
            *list(vgg19_model.features.children())[:18]
        )

    def forward(self, x):
        return self.feature_extractor(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_features, 0.8),
            nn.ReLU(),
            nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=5, scale_factor=2):
        super(GeneratorResNet, self).__init__()
        up_sample_block_num = int(math.log(scale_factor, 2))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 9, 1, 4),
            nn.PReLU(),
        )

        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64, 0.8),
        )

        up_sampling = []
        for _ in range(up_sample_block_num):
            up_sampling += [
                nn.Conv2d(64, 64 * scale_factor ** 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(64 * scale_factor ** 2),
                nn.PixelShuffle(upscale_factor=scale_factor),
                nn.PReLU(),
            ]
        self.up_sampling = nn.Sequential(*up_sampling)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, out_channels, 9, 1, 4),
            nn.Tanh(),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.up_sampling(out)
        out = self.conv3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels

        layers = []
        in_filters = self.in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(
                self.discriminator_block(in_filters, out_filters, first_block=(i == 0))
            )
            in_filters = out_filters
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Conv2d(in_filters, 1024, 1))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(1024, 1, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    @staticmethod
    def discriminator_block(in_filters, out_filters, first_block=False):
        layers = [nn.Conv2d(in_filters, out_filters, 3, 1, 1, bias=False)]
        if not first_block:
            layers.append(nn.BatchNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(out_filters, out_filters, 3, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.model(x).view(batch_size))


if __name__ == "__main__":
    disc_test_input = torch.randn(size=(10, 3, 256, 256)).to(get_device())
    discriminator_model = Discriminator().to(get_device())
    print(discriminator_model)
    disc_output = discriminator_model(disc_test_input)
    print(disc_output.shape)
    print("=" * 100)
    gen_test_input = torch.randn(size=(10, 3, 64, 64)).to(get_device())
    generator_model = GeneratorResNet().to(get_device())
    print(generator_model)
    gen_output = generator_model(gen_test_input)
    print(gen_output.shape)
