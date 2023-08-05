from math import log10
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as torch_functional
from torchvision.utils import make_grid, save_image


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_np(x):
    return x.data.cpu().numpy()


def plot_train_result(
        real_image,
        gen_image,
        recon_image,
        epoch=1,
        save=False,
        show=False,
        fig_size=(8, 2),
):
    fig, axes = plt.subplots(1, 3, figsize=fig_size)
    images = [
        to_np(real_image),
        to_np(gen_image),
        to_np(recon_image)
    ]
    for ax, img in zip(axes.flatten(), images):
        ax.axis("off")
        img = img.squeeze()
        img = (
            (((img - img.min()) * 255) / (img.max() - img.min()))
            .transpose(1, 2, 0)
            .astype(np.uint8)
        )
        ax.imshow(img, cmap=None, aspect="equal")
    plt.subplots_adjust(wspace=0, hspace=0)

    title = "epoch {0}".format(epoch)
    fig.text(0.5, 0.04, title, ha="center")
    if save:
        save_fn = "result_epoch_{:d}".format(epoch + 1) + ".png"
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def train_val(
        epochs,
        train_loader,
        val_loader,
        generator,
        discriminator,
        optimizer_gen,
        optimizer_disc,
        criterion_gen,
        device,
        up_scale_factor,
):
    final_result = {'train': {}, 'val': {}}
    for epoch in range(1, epochs+1):
        print(f"total train batches: {len(train_loader)}")
        train_bar = tqdm(train_loader)
        train_running_results = {'total_samples': 0, 'disc_loss': 0, 'gen_loss': 0, 'd_score': 0, 'g_score': 0}

        for batch_idx, images in enumerate(train_bar, 1):

            generator.train()
            discriminator.train()

            images_lr = Variable(images["lr"]).to(device)
            images_hr = Variable(images["hr"]).to(device)

            batch_size = images_lr.shape[0]
            train_running_results['total_samples'] += batch_size

            gen_hr = generator(images_lr)
            optimizer_disc.zero_grad()
            real_loss = discriminator(images_hr).mean()
            fake_loss = discriminator(gen_hr).mean()
            disc_loss = 1 - real_loss + fake_loss
            disc_loss.backward(retain_graph=True)
            optimizer_disc.step()

            optimizer_gen.zero_grad()
            gen_hr = generator(images_lr)
            fake_loss = discriminator(gen_hr).mean()
            gen_loss = criterion_gen(fake_loss, gen_hr, images_hr)
            gen_loss.backward()
            optimizer_gen.step()

            train_running_results['gen_loss'] += gen_loss.item() * batch_size
            train_running_results['disc_loss'] += disc_loss.item() * batch_size
            train_running_results['d_score'] += real_loss.item() * batch_size
            train_running_results['g_score'] += fake_loss.item() * batch_size

            if batch_idx % int(0.1 * len(train_loader)) == 0:
                plot_train_result(images_hr[0], images_lr[0], gen_hr[0], epoch)

            train_bar.set_description(
                desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                    epoch,
                    epochs,
                    train_running_results['disc_loss'] / train_running_results['total_samples'],
                    train_running_results['gen_loss'] / train_running_results['total_samples'],
                    train_running_results['d_score'] / train_running_results['total_samples'],
                    train_running_results['g_score'] / train_running_results['total_samples']
                )
            )

        final_result['train'][epoch] = train_running_results

        with torch.inference_mode():
            generator.eval()
            discriminator.eval()

            val_bar = tqdm(val_loader)
            val_running_results = {'total_samples': 0, 'mse': 0, 'psnr': 0}
            print(f"total validation batches: {len(val_loader)}")

            for batch_idx, images in enumerate(val_bar, 1):
                generator.train()
                discriminator.train()

                images_lr = Variable(images["lr"]).to(device)
                images_hr = Variable(images["hr"]).to(device)

                batch_size = images_lr.shape[0]
                val_running_results['total_samples'] += batch_size

                gen_hr = generator(images_lr)
                mse_loss = ((gen_hr - images_hr) ** 2).data.mean()
                psnr = 10 * log10((images_hr.max() ** 2) / mse_loss)

                val_running_results['mse'] += mse_loss * batch_size
                val_running_results['psnr'] += psnr * batch_size

                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB ' % (
                            val_running_results['psnr'] / val_running_results['total_samples'])
                )

                if batch_idx % int(0.1 * len(val_loader)) == 0:
                    images_lr = torch_functional.interpolate(images_lr, scale_factor=up_scale_factor)
                    images_hr = make_grid(images_hr, nrow=1, normalize=True)
                    gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
                    images_lr = make_grid(images_lr, nrow=1, normalize=True)
                    image_grid = torch.cat((images_hr, images_lr, gen_hr), -1)
                    save_image(image_grid, f"images/val_{epoch}_{batch_idx}.png", normalize=False)

            final_result['val'][epoch] = val_running_results
            torch.save(generator, "saved_models/generator.pth")
            torch.save(discriminator, "saved_models/discriminator.pth")
