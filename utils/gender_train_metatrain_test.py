import argparse
import random

import numpy as np


def get_classes_count(array):
    array = np.sort(array.astype("int"))
    classes_dict = {}
    classes, classes_count = np.unique(array, return_counts=True)

    for i in range(len(classes)):
        classes_dict[classes[i]] = classes_count[i]

    return classes_dict


def get_random_indices(array, qtd):
    random_indices = np.asarray(
        random.sample(
            list(np.linspace(0, len(array) - 1, len(array))),
            qtd,
        )
    ).astype("int")

    return random_indices


def filter_images_race_gender(image_paths, race, gender):
    indices = np.where(
        (image_paths[:, 1].astype(int) == race)
        & (image_paths[:, 2].astype(int) == gender)
    )[0]

    return indices


def get_metatrain_test(image_paths, race, gender):
    indices = filter_images_race_gender(image_paths, race, gender)

    meta_indices = get_random_indices(indices, 50)
    meta_paths = image_paths[indices[meta_indices]]

    image_paths = np.delete(image_paths, indices[meta_indices], axis=0)

    indices = filter_images_race_gender(image_paths, race, gender)

    test_indices = get_random_indices(indices, 250)
    test_paths = image_paths[indices[test_indices]]

    image_paths = np.delete(image_paths, indices[test_indices], axis=0)

    return image_paths, meta_paths, test_paths


def split(image_list):
    image_paths = np.loadtxt(image_list, dtype=np.str)
    np.random.shuffle(image_paths)

    image_paths = image_paths[
        (image_paths[:, 3].astype(float) >= 18)
        & (image_paths[:, 3].astype(float) < 100)
    ]

    image_paths, wf_meta_train, wf_test = get_metatrain_test(image_paths, 0, 0)
    image_paths, wm_meta_train, wm_test = get_metatrain_test(image_paths, 0, 1)
    image_paths, bf_meta_train, bf_test = get_metatrain_test(image_paths, 1, 0)
    image_paths, bm_meta_train, bm_test = get_metatrain_test(image_paths, 1, 1)

    train_output = image_list[:-4] + "_train.txt"
    meta_train_output = image_list[:-4] + "_meta_train.txt"
    val_output = image_list[:-4] + "_val.txt"

    train_set = image_paths
    meta_train_set = np.concatenate(
        [wf_meta_train, wm_meta_train, bf_meta_train, bm_meta_train]
    )
    test_set = np.concatenate([wf_test, wm_test, bf_test, bm_test])

    print(f"Races in train: {get_classes_count(train_set[:, 1])}")
    print(f"Gender in train: {get_classes_count(train_set[:, 2])}")
    print(f"Age in train: {get_classes_count(train_set[:, 3])}")

    print(f"Races in meta-train: {get_classes_count(meta_train_set[:, 1])}")
    print(f"Gender in meta-train: {get_classes_count(meta_train_set[:, 2])}")
    print(f"Age in meta-train: {get_classes_count(meta_train_set[:, 3])}")

    print(f"Races in test: {get_classes_count(test_set[:, 1])}")
    print(f"Gender in test: {get_classes_count(test_set[:, 2])}")
    print(f"Age in test: {get_classes_count(test_set[:, 3])}")

    np.savetxt(train_output, train_set, fmt="%s")
    np.savetxt(meta_train_output, meta_train_set, fmt="%s")
    np.savetxt(val_output, test_set, fmt="%s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train and test splits.")
    parser.add_argument("-i", "--image_list", help="List of images to split.")

    args = parser.parse_args()

    np.random.seed(0)
    random.seed(0)

    split(args.image_list)
