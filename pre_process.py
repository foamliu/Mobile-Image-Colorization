import os

from config import image_folder, num_train


def split_data():
    for split in ['train', 'val', 'test']:
        folder = 'data/{}'.format(split)
        files = [f for f in os.listdir(folder) if f.lower().endswith('.jpeg')]

        print('num_{}: {}'.format(split, len(files)))

        # train = names[:num_train]
        # valid = names[num_train:]
        #
        # with open('train.txt', 'w') as file:
        #     file.write('\n'.join(train))
        #
        # with open('valid.txt', 'w') as file:
        #     file.write('\n'.join(valid))
        #
        # print('num_train: ' + str(len(train)))
        # print('num_valid: ' + str(len(valid)))


if __name__ == '__main__':
    split_data()
