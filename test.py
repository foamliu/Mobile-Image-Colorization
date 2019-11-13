import numpy as np
import sklearn.neighbors as nn
import cv2 as cv
from data_gen import get_soft_encoding


if __name__ == '__main__':
    nb_neighbors = 5
    # Load the array of quantized ab value
    q_ab = np.load("data/pts_in_hull.npy")
    nb_q = q_ab.shape[0]
    # Fit a NN to q_ab
    nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, metric='euclidean', algorithm='ball_tree').fit(q_ab)

    filename = 'images/0_gt.png'
    # b: 0 <=b<=255, g: 0 <=g<=255, r: 0 <=r<=255.
    bgr = cv.imread(filename)
    bgr = cv.resize(bgr, (2, 2))
    # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
    y = get_soft_encoding(lab[:, :, 1:], nn_finder, nb_q)
    print(y.shape)
    print(y)
    print(y.min())
    print(y.max())

