from os import path

import lmdb
import msgpack
import numpy as np
import six
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils.image_utils import blur_image


class LMDB(Dataset):
    def __init__(self, db_path, transform=None, target_transform=None, use_mask=False):
        self.db_path = db_path
        self.env = lmdb.open(
            db_path,
            subdir=path.isdir(db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.env.begin(write=False) as txn:
            self.length = msgpack.loads(txn.get(b"__len__"))
            self.keys = msgpack.loads(txn.get(b"__keys__"))
            self.classnum = msgpack.loads(txn.get(b"__classnum__"))

        self.transform = transform
        self.target_transform = target_transform
        self.use_mask = use_mask

    def __getitem__(self, index):
        img, target, mask = None, None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = msgpack.loads(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        if self.use_mask:
            imgbuf = unpacked[2]
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            mask = Image.open(buf)
            img = blur_image(np.asarray(img).copy(), np.asarray(mask).copy())
            img = Image.fromarray(img)

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target, self.classnum)

        return img, target

    def _get_label(self, index):
        target = None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = msgpack.loads(byteflow)
        target = unpacked[1]

        return target

    def __len__(self):
        return self.length

    def get_targets(self):
        targets = []
        for idx in range(self.length):
            target = self._get_label(idx)
            targets.append(target)

        return np.asarray(targets)


class LMDBDataLoader(DataLoader):
    def __init__(self, config, lmdb_path, train=True, use_mask=False, meta_train=False):
        transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.RandomHorizontalFlip(0.5 if train else 0),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        target_transform = None

        if config.attribute == "age":
            target_transform = self.transform_ages_to_one_hot_ordinal

        self._dataset = LMDB(lmdb_path, transform, target_transform, use_mask)

        if meta_train:
            batch_size = len(self._dataset)
        else:
            batch_size = config.batch_size

        super(LMDBDataLoader, self).__init__(
            self._dataset,
            batch_size=batch_size,
            shuffle=train,
            pin_memory=config.pin_memory,
            num_workers=config.workers,
            drop_last=train,
        )

    def class_num(self):
        return self._dataset.classnum

    def transform_ages_to_one_hot_ordinal(self, target, classes):
        new_target = np.zeros(shape=classes)
        new_target[:target] = 1

        return new_target.astype("float32")
