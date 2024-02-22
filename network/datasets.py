import os
import numpy as np
from torch.utils.data import Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image


class TrainDataset(Dataset):
    my_transform = transforms.Compose([
        transforms.ToTensor,
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def __init__(self, folder_path, my_transforms=my_transform):
        """
        Main dataset for training neural network (autoencoder)

        :param folder_path: Path to folder with images
        :param my_transforms: Torchvision transforms
        """
        self.data = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                path = os.path.join(root, file)
                img = np.asarray(Image.open(path).convert("RGB"))
                image_array = np.array(img, "uint8")
                self.data.append(image_array)

        self.my_transform = my_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]

        if self.my_transform is not None:
            try:
                data = self.my_transform(data)
            except:
                pass

        return data


class WorkDataset(Dataset):
    my_transform = transforms.Compose([
        transforms.ToTensor,
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def __init__(self, folder_path, my_transform=my_transform):
        """
        Dataset for working neural network (autoencoder)

        :param folder_path: Path to folder with images
        :param my_transform: Torchvision transforms
        """
        self.images = []
        self.data = []

        for root, _, files in os.walk(folder_path):
            for file in files:
                path = os.path.join(root, file)
                img = np.asarray(Image.open(path).convert('RGB'))
                image_array = np.array(img, "uint8")
                self.images.append(img)
                self.data.append([image_array, root.split("/")[-1]])

        self.my_transform = my_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        if self.transforms is not None:
            try:
                data[0] = self.transforms(data[0])
            except:
                pass

        return data


def divide_dataset(dataset: TrainDataset):
    """
    Function for dividing your dataset into training and validation sets

    :param dataset: Your dataset
    :return: Array [training_data, validation_data]
    """
    training_length = len(dataset) * 0.9
    validation_length = len(dataset) - training_length
    training_data, validation_data = random_split(dataset, [training_length, validation_length])
    return [training_data, validation_data]
