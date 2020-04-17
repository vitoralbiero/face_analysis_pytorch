from os import path
import lmdb
import pyarrow as pa
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import lz4framed


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    return lz4framed.compress(pa.serialize(obj).to_buffer())


def list2lmdb(attribute, source, dest, num_workers=16, write_frequency=5000):
    print("Loading dataset from %s" % source)
    dataset = ImageFolder(source, loader=raw_reader)
    data_loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x)

    name = f"{path.split(source)[1][:-4]}.lmdb"
    lmdb_path = path.join(dest, name)
    isdir = path.isdir(lmdb_path)

    print(f"Generate LMDB to {lmdb_path}")

    image_size = 112
    size = len(data_loader.dataset) * image_size * image_size * 3
    print(f'LMDB max size: {size}')

    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=size * 2, readonly=False,
                   meminit=False, map_async=True)

    print(len(data_loader.dataset))
    txn = db.begin(write=True)
    max_label = 0
    for idx, data in tqdm(enumerate(data_loader)):
        # print(type(data), data)
        image, label = data[0]
        max_label = max(max_label, label)
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))
        txn.put(b'__classnum__', dumps_pyarrow(max_label))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_list', '-l', help='List of images.')
    parser.add_argument('--source', '-s', help='Path to the images.')
    parser.add_argument('--workers', '-w', help='Workers number.', default=16, type=int)
    parser.add_argument('--attribute', '-a',
                        help='Which attribute to load [race, gender, age, recognition].', type=str)
    parser.add_argument('--dest', '-d', help='Path to save the lmdb file.')

    args = parser.parse_args()

    list2lmdb(args.attribute, args.source, args.image_list, args.dest, args.workers)
