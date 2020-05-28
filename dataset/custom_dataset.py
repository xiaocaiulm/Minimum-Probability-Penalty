from __future__ import print_function, division
from PIL import Image
from torch.utils.data import Dataset
import os


class CustomDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None, training=False):
        self.image_list = []
        self.id_list = []
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = 0
        self.training = training
        with open(txt_file, 'r') as f:
            line = f.readline()
            while line:
                img_name = line.split()[0]
                label = int(line.split()[1])
                self.image_list.append(img_name)
                self.id_list.append(label)
                line = f.readline()
        self.num_classes = max(self.id_list) + 1

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label = self.id_list[idx]
        img_name = os.path.join(self.root_dir, img_name)
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, label
