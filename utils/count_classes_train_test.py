import numpy as np
import argparse
import random

ATTRIBUTES = {'race': 1, 'gender': 2, 'age': 3}


def get_classes_count(array):
    array = np.sort(array.astype('int'))
    classes_dict = {}
    classes, classes_count = np.unique(array, return_counts=True)

    for i in range(len(classes)):
        classes_dict[classes[i]] = classes_count[i]

    return classes_dict


def split(train_list, test_list, attribute):
    train_set = np.loadtxt(train_list, dtype=np.str)
    # train_set = train_set[train_set[:, ATTRIBUTES[attribute]].astype('int') >= 0]
    # train_set = train_set[train_set[:, ATTRIBUTES[attribute]].astype('int') <= 100]

    test_set = np.loadtxt(test_list, dtype=np.str)
    # test_set = test_set[test_set[:, ATTRIBUTES[attribute]].astype('int') >= 0]
    # test_set = test_set[test_set[:, ATTRIBUTES[attribute]].astype('int') <= 100]

    print(f'{attribute} in train: {get_classes_count(train_set[:, ATTRIBUTES[attribute]])}')
    print('')
    print(f'{attribute} in train: {get_classes_count(test_set[:, ATTRIBUTES[attribute]])}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Count number of classes in train and test sets.')
    parser.add_argument('-t', '--train_list', help='List of images to split.')
    parser.add_argument('-v', '--test_list', help='List of images to split.')
    parser.add_argument('-a', '--attribute', help='Attribute to count. [race, gender, age]', type=str)

    args = parser.parse_args()

    np.random.seed(0)
    random.seed(0)

    split(args.train_list, args.test_list, args.attribute.lower())
