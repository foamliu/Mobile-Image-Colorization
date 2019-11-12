import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

im_size = 256
channel = 3
batch_size = 32
epochs = 10000
patience = 50
num_train_samples = 529202
num_valid_samples = 4268
num_classes = 313
kernel = 3
weight_decay = 1e-3
epsilon = 1e-8
nb_neighbors = 5
# temperature parameter T
T = 0.38
