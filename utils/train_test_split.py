import numpy as np
import argparse
import random


def get_classes_count(array):
    array = np.sort(array.astype('int'))
    classes_dict = {}
    classes, classes_count = np.unique(array, return_counts=True)

    for i in range(len(classes)):
        classes_dict[classes[i]] = classes_count[i]

    return classes_dict


def split(image_list, percent):
    image_paths = np.loadtxt(image_list, dtype=np.str)
    np.random.shuffle(image_paths)

    indices = np.asarray(random.sample(list(np.linspace(0, len(image_paths) - 1, len(image_paths))),
                                       int(percent * len(image_paths)))).astype('int')

    train_set = np.delete(image_paths, indices, axis=0)
    test_set = image_paths[indices]

    assert len(train_set) + len(test_set) == len(image_paths)

    train_output = image_list[:-4] + '_train.txt'
    val_output = image_list[:-4] + '_val.txt'

    print(f'Races in train: {get_classes_count(train_set[:, 1])}')
    print(f'Gender in train: {get_classes_count(train_set[:, 2])}')
    print(f'Age in train: {get_classes_count(train_set[:, 3])}')

    print(f'Races in test: {get_classes_count(test_set[:, 1])}')
    print(f'Gender in test: {get_classes_count(test_set[:, 2])}')
    print(f'Age in test: {get_classes_count(test_set[:, 3])}')

    np.savetxt(train_output, train_set, fmt="%s")
    np.savetxt(val_output, test_set, fmt="%s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create train and test splits.')
    parser.add_argument('-i', '--image_list', help='List of images to split.')
    parser.add_argument('-p', '--percent', help='Percent of data used to test.', default=0.1)

    args = parser.parse_args()

    np.random.seed(0)
    random.seed(0)

    split(args.image_list, args.percent)
