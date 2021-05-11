import argparse
import pickle
from os import path

import bcolz
import mxnet as mx
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
)


def load_bin(path, image_size=[112, 112]):
    rootdir = path[:-4]

    bins, issame_list = pickle.load(open(path, "rb"), encoding="bytes")
    data = bcolz.fill(
        [len(bins), 3, image_size[0], image_size[1]],
        dtype=np.float32,
        rootdir=rootdir,
        mode="w",
    )

    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = TRANSFORM(img)

        if i % 1000 == 0:
            print("Loading bin...", i)

    print(data.shape)
    np.save(str(rootdir) + "_list", np.array(issame_list))

    return data, issame_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="for face verification")
    parser.add_argument(
        "-r", "--rec_path", help="Path to test sets bin files", type=str
    )
    args = parser.parse_args()

    bin_files = ["agedb_30", "cfp_fp", "lfw"]

    for i in range(len(bin_files)):
        load_bin(path.join(args.rec_path, bin_files[i] + ".bin"))
