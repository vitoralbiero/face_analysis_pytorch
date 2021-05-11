import argparse
import random

import numpy as np

ATTRIBUTES = {"race": 1, "gender": 2, "age": 3}


def get_classes_count(array):
    array = np.sort(array.astype("int"))
    classes_dict = {}
    classes, classes_count = np.unique(array, return_counts=True)

    for i in range(len(classes)):
        classes_dict[classes[i]] = classes_count[i]

    return classes_dict


def split(image_list, attribute):
    image_set = np.loadtxt(image_list, dtype=np.str)

    print(
        f"{attribute} in list: {get_classes_count(image_set[:, ATTRIBUTES[attribute]])}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count number of classes in datasets.")
    parser.add_argument("-i", "--image_list", help="List of images to split.")
    parser.add_argument(
        "-a", "--attribute", help="Attribute to count. [race, gender, age]", type=str
    )

    args = parser.parse_args()

    np.random.seed(0)
    random.seed(0)

    split(args.image_list, args.attribute.lower())
