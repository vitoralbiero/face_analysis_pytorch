import argparse
from os import path

import numpy as np
from tqdm import tqdm

# males = 1, females = 0
# white = 0, black = 1, asian = 2, indian = 3,
#   other (e.g., hispanic, latino, middle eastern) = 4

MORPH_RACES = {"W": 0, "B": 1, "A": 2, "I": 3, "H": 4, "O": 4, "U": -1}


def AAF(image_path):
    image_name = path.split(image_path[0])[1][:-4]
    age = int(image_name.split("A")[1])
    gender = int(image_path[1])
    image_path = image_path[0]

    return image_path, gender, age


def AFAD(image_path):
    race = 2
    age = int(image_path.split("/")[-3])

    if int(image_path.split("/")[-2]) == 111:
        gender = 1
    elif int(image_path.split("/")[-2]) == 112:
        gender = 0

    return race, gender, age


def AgeDB(image_path):
    image_name = path.split(image_path)[1][:-4]

    age = int(image_name.split("_")[-2])

    if image_name.split("_")[-1] == "m":
        gender = 1
    elif image_name.split("_")[-1] == "f":
        gender = 0

    return gender, age


def CACD(image_path):
    image_name = path.split(image_path)[1][:-4]
    age = int(image_name.split("_")[0])

    return age


def IMDB_WIKI(image_path):
    if image_path[1] == "nan":
        gender = -1
    else:
        gender = int(image_path[1])

    age = int(image_path[2])

    return image_path[0], gender, age


def IMFDB(image_path):
    race = 3
    if image_path[1] == "MALE":
        gender = 1
    elif image_path[1] == "FEMALE":
        gender = 0

    return image_path[0], race, gender


def MegaAgeAsian(image_path):
    race = 2
    age = int(image_path[1])

    return image_path[0], race, age


def MORPH3(image_path):
    race = int(MORPH_RACES[image_path[1]])

    if image_path[2] == "M":
        gender = 1
    elif image_path[2] == "F":
        gender = 0

    age = int(image_path[3])

    return image_path[0], race, gender, age


def UTKFace(image_path):
    image_name = path.split(image_path)[1]
    age, gender, race = image_name.split("_")[0:3]

    if int(gender) == 1:
        gender = 0

    elif int(gender) == 0:
        gender = 1

    return image_path, int(race), int(gender), int(age)


def normalize_annotations(images_path, dataset_name):
    images = np.loadtxt(images_path, dtype=np.str)
    output_paths = []

    for image_path in tqdm(images):
        race = -1
        gender = -1
        age = -1

        if dataset_name == "AAF":
            image_path, gender, age = AAF(image_path)

        elif dataset_name == "AFAD":
            race, gender, age = AFAD(image_path)

        elif dataset_name == "AGEDB":
            gender, age = AgeDB(image_path)

        elif dataset_name == "CACD":
            age = CACD(image_path)

        elif dataset_name == "IMDB" or dataset_name == "WIKI":
            image_path, gender, age = IMDB_WIKI(image_path)

        elif dataset_name == "IMFDB":
            image_path, race, gender = IMFDB(image_path)

        elif dataset_name == "MEGAAGEASIAN":
            image_path, race, age = MegaAgeAsian(image_path)

        elif dataset_name == "MORPH3":
            image_path, race, gender, age = MORPH3(image_path)

        elif dataset_name == "UTKFACE":
            image_path, race, gender, age = UTKFace(image_path)

        else:
            raise Exception("NO FILE PATTERN FOR THE DATASET INFORMED.")

        output_paths.append([image_path, race, gender, age])

    output = images_path[:-4] + "_normalized_annotations.txt"

    np.savetxt(output, output_paths, fmt="%s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset in half using subjects."
    )
    parser.add_argument("--images_path", "-i", help="File with a list of images.")
    parser.add_argument("--dataset_name", "-d", help="Dataset name.")

    args = parser.parse_args()

    normalize_annotations(args.images_path, args.dataset_name)
