import numpy as np

from utils import get_device

load_pretrained_model = True
train_batch_size = 4
val_batch_size = 2
learning_rate = 1e-3
dataset_path = "./img_align_celeb"
num_cpu_cores = 6
num_channels = 3
b1 = 0.5
b2 = 0.999
device = get_device()
data_mean = np.array([0.485, 0.456, 0.406])
data_std = np.array([0.229, 0.224, 0.225])
