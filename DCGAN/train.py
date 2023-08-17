import random

import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from torch import nn

from models import *
from consts import *
from tqdm import tqdm
from utils import *
from torchvision.utils import make_grid


LUCKY_NUMBER = 47
DEVICE = get_device()
BATCH_SIZE = params["batch_size"]
random.seed(LUCKY_NUMBER)
torch.manual_seed(LUCKY_NUMBER)


dataloader = get_celeb(params)

generator_network = Generator(params).to(DEVICE)
generator_network.apply(weights_init)
generator_optimizer = torch.optim.Adam(generator_network.parameters(), lr=params["lr"])

discriminator_network = Discriminator(params).to(DEVICE)
discriminator_network.apply(weights_init)
discriminator_optimizer = torch.optim.Adam(
    discriminator_network.parameters(), lr=params["lr"]
)

criterion = nn.BCELoss()
noise_input = torch.randn(size=(BATCH_SIZE, params["nz"], 1, 1), device=DEVICE)

img_list = []
gen_losses = []
disc_losses = []
iters = 0

print("Starting Training Loop...")
print("-" * 25)

for epoch in range(params["num_epochs"]):
    train_bar = tqdm(dataloader)
    description = None
    for i, data in enumerate(train_bar, 1):
        real_data = data[0].to(DEVICE)
        discriminator_network.zero_grad()
        label = torch.full(
            (BATCH_SIZE,), real_label, device=DEVICE, dtype=torch.float32
        )
        real_output = discriminator_network(real_data).view(-1)
        disc_real_loss = criterion(real_output, label)
        disc_real_loss.backward()
        noise_input = torch.randn(
            size=(BATCH_SIZE, params["nz"], 1, 1), device=DEVICE, dtype=torch.float32
        )
        fake_data = generator_network(noise_input)
        label.fill_(fake_label)
        fake_output = discriminator_network(fake_data.detach()).view(-1)
        disc_fake_loss = criterion(fake_output, label)
        disc_fake_loss.backward()
        disc_loss = disc_fake_loss + disc_real_loss
        discriminator_optimizer.step()

        generator_network.zero_grad()
        label.fill_(real_label)
        gen_output = discriminator_network(fake_data).view(-1)
        gen_loss = criterion(gen_output, label)
        gen_loss.backward()
        generator_optimizer.step()

        gen_losses.append(gen_loss.item())
        disc_losses.append(disc_loss.item())

        description = f"[{epoch+1}/{params['num_epochs']}] | Disc Real Loss: {disc_real_loss.item():.4f} | Disc Fake Loss: {disc_fake_loss.item():.4f} | Disc Loss: {disc_loss.item():.4f} | Gen Loss: {gen_loss:.4f} | D(G(z)): {fake_output.mean().item():.4f}/{gen_output.mean().item():.4f}"
        train_bar.set_description(desc=description)

        if i % 100 == 0:
            with torch.inference_mode():
                fake_data = generator_network(noise_input).detach().cpu()
            img_list.append(make_grid(fake_data, padding=2, normalize=True))

    print(description)

    if epoch % params["save_epoch"] == 0:
        torch.save(
            {
                "generator": generator_network,
                "discriminator": discriminator_network,
                "generator_optimizer": generator_optimizer,
                "discriminator_optimizer": discriminator_optimizer,
                "params": params,
            },
            f"saved_models/model_epoch_{epoch+1}.pth",
        )

torch.save(
    {
        "generator": generator_network,
        "discriminator": discriminator_network,
        "generator_optimizer": generator_optimizer,
        "discriminator_optimizer": discriminator_optimizer,
        "params": params,
    },
    f"saved_models/model_final.pth",
)


plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(gen_losses, label="G")
plt.plot(disc_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.show()
anim.save("celeb.gif", dpi=80, writer="imagemagick")
