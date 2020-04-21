from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from os import path
import six
import lmdb
import pyarrow as pa
import lz4framed
import numpy as np


class LMDB(Dataset):
    def __init__(self, db_path, transform=None, target_transform=None, train=True):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(lz4framed.decompress(txn.get(b'__len__')))
            self.keys = pa.deserialize(lz4framed.decompress(txn.get(b'__keys__')))
            self.classnum = pa.deserialize(lz4framed.decompress(txn.get(b'__classnum__')))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(lz4framed.decompress(byteflow))

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target, self.classnum)

        return img, target

    def __len__(self):
        return self.length


class LMDBDataLoader(DataLoader):
    def __init__(self, config, lmdb_path, train=True):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5 if train else 0),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        target_transform = None

        if config.attribute == 'age':
            target_transform = self.transform_ages_to_one_hot_ordinal

        self._dataset = LMDB(lmdb_path, transform, target_transform, train)

        super(LMDBDataLoader, self).__init__(self._dataset, batch_size=config.batch_size, shuffle=train,
                                             pin_memory=config.pin_memory, num_workers=config.workers,
                                             drop_last=train)

    def class_num(self):
        return self._dataset.classnum

    def transform_ages_to_one_hot_ordinal(self, target, classes):
        new_target = np.zeros(shape=classes)
        new_target[:target] = 1

        return new_target.astype('float32')
