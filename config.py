import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

im_size = 256
channel = 3
batch_size = 32
epochs = 10000

# Training parameters
num_workers = 1  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

num_train_samples = 529202
num_valid_samples = 4268
num_classes = 313
kernel = 3
epsilon = 1e-8
nb_neighbors = 5
# temperature parameter T
T = 0.38

image_folder = 'data/train2017'
