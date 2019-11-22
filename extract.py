import os
import tarfile


def extract(filename, split):
    print('Extracting {}...'.format(filename))
    tar = tarfile.open(filename)
    folder = 'data/{}'.format(split)
    os.makedirs(folder, exist_ok=True)
    tar.extractall(folder)
    tar.close()


if __name__ == "__main__":
    for split in ['test', 'val', 'train']:
        filename = 'data/ILSVRC2012_img_{}.tar'.format(split)
        extract(filename, split)
