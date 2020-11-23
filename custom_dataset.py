import os
import torch
import random
from PIL import Image


class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        def get_images(class_name):
            images = [x for x in os.listdir(
                image_dirs[class_name]) if x[-3:].lower().endswith('png')]
            print(f'found {len(images)} {class_name} examples.')
            return images

        self.images = {}
        self.classes = ['covid', 'normal', 'viral']

        for class_name in self.classes:
            self.images[class_name] = get_images(class_name)

        self.image_dirs = image_dirs
        self.transform = transform

    def __len__(self):
        return sum(
            [len(self.images[class_name]) for class_name in self.classes]
        )

    def __getitem__(self, index):
        class_name = random.choice(self.classes)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.classes.index(class_name)
