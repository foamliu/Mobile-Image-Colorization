import tarfile


def extract(filename):
    print('Extracting {}...'.format(filename))
    tar = tarfile.open(filename)
    tar.extractall('data')
    tar.close()


if __name__ == "__main__":
    for filename in ['data/ILSVRC2012_img_train.tar', 'data/ILSVRC2012_img_val.tar', 'data/ILSVRC2012_img_test.tar']:
        extract(filename)
