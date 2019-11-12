import os
import random

import numpy as np

from config import image_folder


def split_data():
    names = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]

    num_samples = len(names)  # 1341430
    print('num_samples: ' + str(num_samples))

    num_train_samples = int(num_samples * 0.992)
    print('num_train_samples: ' + str(num_train_samples))
    num_valid_samples = num_samples - num_train_samples
    print('num_valid_samples: ' + str(num_valid_samples))
    valid_names = random.sample(names, num_valid_samples)
    train_names = [n for n in names if n not in valid_names]
    np.random.shuffle(valid_names)
    np.random.shuffle(train_names)

    # with open('names.txt', 'w') as file:
    #     file.write('\n'.join(names))

    with open('valid_names.txt', 'w') as file:
        file.write('\n'.join(valid_names))

    with open('train_names.txt', 'w') as file:
        file.write('\n'.join(train_names))


if __name__ == '__main__':
    split_data()
