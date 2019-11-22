import os

import cv2 as cv
import numpy as np
import sklearn.neighbors as nn
import torch
from torch.utils.data import Dataset

from config import im_size, nb_neighbors


def get_soft_encoding(image_ab, nn_finder, nb_q):
    h, w = image_ab.shape[:2]
    a = np.ravel(image_ab[:, :, 0])
    b = np.ravel(image_ab[:, :, 1])
    ab = np.vstack((a, b)).T
    # Get the distance to and the idx of the nearest neighbors
    dist_neighb, idx_neigh = nn_finder.kneighbors(ab)
    # Smooth the weights with a gaussian kernel
    sigma_neighbor = 5
    wts = np.exp(-dist_neighb ** 2 / (2 * sigma_neighbor ** 2))
    wts = wts / np.sum(wts, axis=1)[:, np.newaxis]
    # format the tar get
    y = np.zeros((ab.shape[0], nb_q))
    idx_pts = np.arange(ab.shape[0])[:, np.newaxis]
    y[idx_pts, idx_neigh] = wts
    y = y.reshape((h, w, nb_q))
    # y = np.argmax(y, axis=2)
    return y


def negative_log_likelihood(y_pred, y_true):
    loss = -y_true * torch.log(y_pred)
    return loss.mean()


class MICDataset(Dataset):
    def __init__(self, split):
        self.split = split

        self.folder = 'data/{}'.format(split)
        self.files = [f for f in os.listdir(self.folder) if f.lower().endswith('.jpeg')]

        # Load the array of quantized ab value
        q_ab = np.load("data/pts_in_hull.npy")
        self.nb_q = q_ab.shape[0]
        # Fit a NN to q_ab
        self.nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

    def __getitem__(self, i):
        name = self.files[i]
        filename = os.path.join(self.folder, name)
        # b: 0<=b<=255, g: 0<=g<=255, r: 0<=r<=255.
        bgr = cv.imread(filename)
        gray = cv.imread(filename, 0)
        gray = cv.resize(gray, (im_size, im_size), cv.INTER_CUBIC)
        lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
        x = gray / 255.

        out_lab = cv.resize(lab, (im_size // 4, im_size // 4), cv.INTER_CUBIC)
        # Before: 42 <=a<= 226, 20 <=b<= 223
        # After: -86 <=a<= 98, -108 <=b<= 95
        out_ab = out_lab[:, :, 1:].astype(np.int32) - 128

        y = get_soft_encoding(out_ab, self.nn_finder, self.nb_q)
        # print('y: ' + str(y))

        if np.random.random_sample() > 0.5:
            x = np.fliplr(x)
            y = np.fliplr(y)

        x = np.expand_dims(x, axis=0)
        x = np.clip(x, 0.0, 1.0)
        x = torch.from_numpy(x)
        y = np.clip(y, 0, float('inf'))
        y = torch.from_numpy(y)
        y = y.transpose(0, 2)

        # print('x.shape: ' + str(x.shape))
        # print('x: ' + str(x))
        # print('y.shape: ' + str(y.shape))
        # print('y: ' + str(y))
        return x, y

    def __len__(self):
        return len(self.files)
