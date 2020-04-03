from os import path
from torchvision import transforms
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import pandas as pd

ATTRIBUTES = {'race': 1, 'gender': 2, 'age': 3, 'recognition': 1}
MAX_CLASS = {'race': 5, 'gender': 2, 'age': 101, 'recognition': float('Inf')}


class ImageList(ImageFolder):
    def __init__(self, config, source, image_list, train=True):
        attribute = ATTRIBUTES[config.attribute]

        image_names = pd.read_csv(image_list, delimiter=' ', header=None)
        image_names = np.asarray(image_names)

        # remove images that have labels outside of desired range
        image_names = image_names[image_names[:, attribute].astype('int') >= 0]
        image_names = image_names[image_names[:, attribute].astype('int') < MAX_CLASS[config.attribute]]

        np.random.shuffle(image_names)

        self.samples = [path.join(source, image_name) for image_name in image_names[:, 0]]

        if config.attribute == 'age':
            self.targets = []
            for output in image_names[:, attribute].astype('int'):
                target = np.zeros(shape=MAX_CLASS['age'] - 1)
                target[:output] = 1
                self.targets.append(target)
            self.targets = np.array(self.targets).astype('float32')

        else:
            self.targets = image_names[:, attribute].astype('int')

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5 if train else 0),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert('RGB')

        return self.transform(img), self.targets[index]


class CustomDataLoader(DataLoader):
    def __init__(self, config, source, image_list, shuffle=True, drop_last=True, train=True):
        self._dataset = ImageList(config, source, image_list, train)

        super(CustomDataLoader, self).__init__(self._dataset, batch_size=config.batch_size, shuffle=shuffle,
                                               pin_memory=config.pin_memory, num_workers=config.workers,
                                               drop_last=drop_last)

    def class_num(self):
        return np.max(self._dataset[: 1]) + 1
