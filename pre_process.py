import os


def split_data():
    for split in ['train', 'val', 'test']:
        folder = 'data/{}'.format(split)
        files = [f for f in os.listdir(folder) if f.lower().endswith('.jpeg')]

        print('num_{}: {}'.format(split, len(files)))


if __name__ == '__main__':
    split_data()
