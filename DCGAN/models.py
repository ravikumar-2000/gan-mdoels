import torch
from torch import nn


def weights_init(layer):
    classname = layer.__class__.__name__
    if classname.find("conv") != -1:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif classname.find("bn") != -1:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.conv_trans1 = nn.ConvTranspose2d(
            params["nz"], params["ngf"] * 8, 4, 1, 0, bias=False
        )
        self.conv_trans2 = nn.ConvTranspose2d(
            params["ngf"] * 8, params["ngf"] * 4, 4, 2, 1, bias=False
        )
        self.conv_trans3 = nn.ConvTranspose2d(
            params["ngf"] * 4, params["ngf"] * 2, 4, 2, 1, bias=False
        )
        self.conv_trans4 = nn.ConvTranspose2d(
            params["ngf"] * 2, params["ngf"], 4, 2, 1, bias=False
        )
        self.conv_trans5 = nn.ConvTranspose2d(
            params["ngf"], params["nc"], 4, 2, 1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(params["ngf"] * 8)
        self.bn2 = nn.BatchNorm2d(params["ngf"] * 4)
        self.bn3 = nn.BatchNorm2d(params["ngf"] * 2)
        self.bn4 = nn.BatchNorm2d(params["ngf"])

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv_trans1(x)))
        x = self.relu(self.bn2(self.conv_trans2(x)))
        x = self.relu(self.bn3(self.conv_trans3(x)))
        x = self.relu(self.bn4(self.conv_trans4(x)))
        x = self.tanh(self.conv_trans5(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(params["nc"], params["nh"], 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(params["nh"], params["nh"] * 2, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(params["nh"] * 2, params["nh"] * 4, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(params["nh"] * 4, params["nh"] * 8, 4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(params["nh"] * 8, 1, 4, 1, 0, bias=False)

        self.bn2 = nn.BatchNorm2d(params["nh"] * 2)
        self.bn3 = nn.BatchNorm2d(params["nh"] * 4)
        self.bn4 = nn.BatchNorm2d(params["nh"] * 8)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = self.sigmoid(self.conv5(x))
        return x


if __name__ == "__main__":
    discriminator_test_data = torch.randn((1, 3, 64, 64))
    discriminator_network = Discriminator({"nc": 3, "nh": 64})
    discriminator_output = discriminator_network(discriminator_test_data)
    print(discriminator_output, discriminator_output.shape)

    print("*" * 100)

    generator_test_data = torch.randn((1, 100, 1, 1))
    generator_network = Generator({"nc": 3, "nz": 100, "ngf": 64})
    generator_output = generator_network(generator_test_data)
    print(generator_output, generator_output.shape)
