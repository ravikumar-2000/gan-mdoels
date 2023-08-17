params = {
    "batch_size": 32,  
    "img_size": 64,  
    "nc": 3,  # Number of channels in the training images. For coloured images this is 3.
    "nz": 100,  # Size of the Z latent vector (the input to the generator).
    "ngf": 64,  # Size of feature maps in the generator. The depth will be multiples of this.
    "nh": 64,  # Size of features maps in the discriminator. The depth will be multiples of this.
    "num_epochs": 100,
    "lr": 0.0002,
    "beta_1": 0.5,
    "save_epoch": 20,
}

data_root_folder = "../data/"
real_label = 1
fake_label = 0

