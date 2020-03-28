from os import path
from torchvision import transforms
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import pandas as pd

ATTRIBUTES = {'race': 1, 'gender': 2, 'age': 3, 'recognition': 1}


class ImageList(ImageFolder):
    def __init__(self, config, source, image_list, train=True):
        attribute = ATTRIBUTES[config.attribute]

        image_names = pd.read_csv(image_list, delimiter=' ', header=None)
        image_names = np.asarray(image_names)

        # remove images with unknown annotation
        image_names = image_names[image_names[:, attribute].astype('int') != -1]

        np.random.shuffle(image_names)

        self.samples = [path.join(source, image_name) for image_name in image_names[:, 0]]
        self.targets = image_names[:, attribute].astype('int')

        if attribute == 3:
            self.targets = self.targets.astype('float')

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
        return self._dataset[-1][1] + 1
