import os

from config import image_folder, num_train


def split_data():
    names = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]

    num_samples = len(names)  # 118287
    print('num_samples: ' + str(num_samples))

    train = names[:num_train]
    valid = names[num_train:]

    with open('train.txt', 'w') as file:
        file.write('\n'.join(train))

    with open('valid.txt', 'w') as file:
        file.write('\n'.join(valid))

    print('num_train: ' + str(len(train)))
    print('num_valid: ' + str(len(valid)))


if __name__ == '__main__':
    split_data()
