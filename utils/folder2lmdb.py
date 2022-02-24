from os import path

import lmdb
import msgpack
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def raw_reader(path):
    with open(path, "rb") as f:
        bin_data = f.read()
    return bin_data


def list2lmdb(source, dest, name, num_workers=16, write_frequency=5000):
    print("Loading dataset from %s" % source)
    dataset = ImageFolder(source, loader=raw_reader)
    data_loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x)

    lmdb_path = path.join(dest, f"{name}.lmdb")
    isdir = path.isdir(lmdb_path)

    print(f"Generate LMDB to {lmdb_path}")

    image_size = 112
    size = len(data_loader.dataset) * image_size * image_size * 3
    print(f"LMDB max size: {size}")

    db = lmdb.open(
        lmdb_path,
        subdir=isdir,
        map_size=size * 2,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    print(len(dataset), len(data_loader))
    txn = db.begin(write=True)
    max_label = 0
    for idx, data in tqdm(enumerate(data_loader)):
        # print(type(data), data)
        image, label = data[0]
        max_label = max(max_label, label)
        txn.put("{}".format(idx).encode("ascii"), msgpack.dumps((image, int(label))))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = ["{}".format(k).encode("ascii") for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", msgpack.dumps(keys))
        txn.put(b"__len__", msgpack.dumps(len(keys)))
        txn.put(b"__classnum__", msgpack.dumps(max_label + 1))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s", help="Path to the images.")
    parser.add_argument("--workers", "-w", help="Workers number.", default=16, type=int)
    parser.add_argument("--dest", "-d", help="Path to save the lmdb file.")
    parser.add_argument("--name", "-n", help="Name for the lmdb file.")

    args = parser.parse_args()

    list2lmdb(args.source, args.dest, args.name, args.workers)
