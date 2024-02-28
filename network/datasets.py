import os
import numpy as np
from torch.utils.data import Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image


class TrainDataset(Dataset):
    def __init__(self, folder_path, transforms=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])):
        """
        Main dataset for training neural network (autoencoder)

        :param folder_path: Path to folder with images
        :param transforms: Torchvision transforms
        """
        self.data = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                path = os.path.join(root, file)
                img = np.asarray(Image.open(path).convert("RGB"))
                image_array = np.array(img, "uint8")
                self.data.append(image_array)

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        if self.transforms is not None:
            try:
                data = self.transforms(data)
            except:
                pass

        return data


class WorkDataset(Dataset):
    def __init__(self, folder_path, transforms=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])):
        """
        Dataset for working neural network (autoencoder)

        :param folder_path: Path to folder with images
        :param transforms: Torchvision transforms
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
        self.transforms = transforms

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
    Function for dividing your dataset into training and validation dataset

    :param dataset: Your dataset
    :param training_length: Length for training dataset
    :return: Array [training_data, validation_data]
    """
    training_length = int(len(dataset) * 0.8)
    validation_length = len(dataset) - training_length
    training_data, validation_data = random_split(dataset, [training_length, validation_length])
    return [training_data, validation_data]
