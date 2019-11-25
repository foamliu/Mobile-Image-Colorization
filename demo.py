# import the necessary packages
import os
import random
from config import device

import cv2 as cv
import numpy as np
import sklearn.neighbors as nn
import torch

from config import im_size, nb_neighbors, T, epsilon

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    print('loading {}...'.format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model'].module
    model = model.to(device)
    model.eval()

    folder = 'data/test'
    files = [f for f in os.listdir(folder) if f.lower().endswith('.jpeg')]
    samples = random.sample(files, 10)

    h, w = im_size // 4, im_size // 4

    # Load the array of quantized ab value
    q_ab = np.load("data/pts_in_hull.npy")
    nb_q = q_ab.shape[0]

    # Fit a NN to q_ab
    nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(folder, image_name)
        print('Start processing image: {}'.format(filename))
        # b: 0 <=b<=255, g: 0 <=g<=255, r: 0 <=r<=255.
        bgr = cv.imread(filename)
        gray = cv.imread(filename, 0)
        bgr = cv.resize(bgr, (im_size, im_size), cv.INTER_CUBIC)
        gray = cv.resize(gray, (im_size, im_size), cv.INTER_CUBIC)
        # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
        lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        a = lab[:, :, 1]
        b = lab[:, :, 2]
        # print('np.max(L): ' + str(np.max(L)))
        # print('np.min(L): ' + str(np.min(L)))
        # print('np.max(a): ' + str(np.max(a)))
        # print('np.min(a): ' + str(np.min(a)))
        # print('np.max(b): ' + str(np.max(b)))
        # print('np.min(b): ' + str(np.min(b)))
        x_test = np.empty((1, 1, im_size, im_size), dtype=np.float32)
        x_test[0, 0, :, :] = gray / 255.
        x_test = torch.from_numpy(x_test)
        x_test = x_test.to(device)

        # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
        with torch.no_grad():
            X_colorized = model(x_test)

        X_colorized = torch.transpose(X_colorized, 1, 3)
        X_colorized = X_colorized.cpu().numpy()
        X_colorized = X_colorized.reshape((h * w, nb_q))

        # Reweight probas
        X_colorized = np.exp(np.log(X_colorized + epsilon) / T)
        X_colorized = X_colorized / np.sum(X_colorized, 1)[:, np.newaxis]

        # Reweighted
        q_a = q_ab[:, 0].reshape((1, 313))
        q_b = q_ab[:, 1].reshape((1, 313))

        X_a = np.sum(X_colorized * q_a, 1).reshape((h, w))
        X_b = np.sum(X_colorized * q_b, 1).reshape((h, w))
        # print('np.max(X_a): ' + str(np.max(X_a)))
        # print('np.min(X_a): ' + str(np.min(X_a)))
        # print('np.max(X_b): ' + str(np.max(X_b)))
        # print('np.min(X_b): ' + str(np.min(X_b)))
        X_a = cv.resize(X_a, (im_size, im_size), cv.INTER_CUBIC)
        X_b = cv.resize(X_b, (im_size, im_size), cv.INTER_CUBIC)

        # Before: -90 <=a<= 100, -110 <=b<= 110
        # After: 38 <=a<= 228, 18 <=b<= 238
        X_a = X_a + 128
        X_b = X_b + 128
        # print('np.max(X_a): ' + str(np.max(X_a)))
        # print('np.min(X_a): ' + str(np.min(X_a)))
        # print('np.max(X_b): ' + str(np.max(X_b)))
        # print('np.min(X_b): ' + str(np.min(X_b)))

        out_lab = np.zeros((im_size, im_size, 3), dtype=np.int32)
        out_lab[:, :, 0] = lab[:, :, 0]
        out_lab[:, :, 1] = X_a
        out_lab[:, :, 2] = X_b
        out_L = out_lab[:, :, 0]
        out_a = out_lab[:, :, 1]
        out_b = out_lab[:, :, 2]
        # print('np.max(out_L): ' + str(np.max(out_L)))
        # print('np.min(out_L): ' + str(np.min(out_L)))
        # print('np.max(out_a): ' + str(np.max(out_a)))
        # print('np.min(out_a): ' + str(np.min(out_a)))
        # print('np.max(out_b): ' + str(np.max(out_b)))
        # print('np.min(out_b): ' + str(np.min(out_b)))
        out_lab = out_lab.astype(np.uint8)
        out_bgr = cv.cvtColor(out_lab, cv.COLOR_LAB2BGR)
        # print('np.max(out_bgr): ' + str(np.max(out_bgr)))
        # print('np.min(out_bgr): ' + str(np.min(out_bgr)))
        out_bgr = out_bgr.astype(np.uint8)

        if not os.path.exists('images'):
            os.makedirs('images')

        cv.imwrite('images/{}_image.png'.format(i), gray)
        cv.imwrite('images/{}_gt.png'.format(i), bgr)
        cv.imwrite('images/{}_out.png'.format(i), out_bgr)
