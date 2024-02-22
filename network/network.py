import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as f
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import cv2
import pandas as pd


class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, latent_dim=1000, act_fn=nn.ReLU()):
        """
        Encoder class for autoencoder

        :param in_channels: Input channels ("RGB" - 3)
        :param out_channels: Output channels
        :param latent_dim: Latent dimension resolution
        :param act_fn: Activation function
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            act_fn,
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            act_fn,
            nn.Conv2d(out_channels, 2 * out_channels, 3, padding=1),
            act_fn,
            nn.Conv2d(2 * out_channels, 2 * out_channels, 3, padding=1),
            act_fn,
            nn.Conv2d(2 * out_channels, 4 * out_channels, 3, padding=1, stride=2),
            act_fn,
            nn.Conv2d(4 * out_channels, 4 * out_channels, 3, padding=1),
            act_fn,
            nn.Conv2d(4 * out_channels, 8 * out_channels, 3, padding=1, stride=2),
            act_fn,
            nn.Conv2d(8 * out_channels, 8 * out_channels, 3, padding=1),
            act_fn,
            nn.Flatten(),
            nn.Linear(8 * out_channels * 14 * 14, latent_dim),
            act_fn
        )

    def forward(self, x):
        x = x.view(-1, 3, 112, 112)
        output = self.net(x)
        return output


class Decoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, latent_dim=1000, act_fn=nn.ReLU()):
        """
        Decoder class for autoencoder

        :param in_channels: Input channels ("RGB" - 3)
        :param out_channels: Output channels
        :param latent_dim: Latent dimension resolution
        :param act_fn: Activate function
        """
        super().__init__()

        self.out_channels = out_channels

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 8 * out_channels * 14 * 14),
            act_fn
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(8 * out_channels, 8 * out_channels, 3, padding=1),
            act_fn,
            nn.ConvTranspose2d(8 * out_channels, 4 * out_channels, 3, padding=1, stride=2, output_padding=1),
            act_fn,
            nn.ConvTranspose2d(4 * out_channels, 4 * out_channels, 3, padding=1),
            act_fn,
            nn.ConvTranspose2d(4 * out_channels, 2 * out_channels, 3, padding=1, stride=2, output_padding=1),
            act_fn,
            nn.ConvTranspose2d(2 * out_channels, 2 * out_channels, 3, padding=1),
            act_fn,
            nn.ConvTranspose2d(2 * out_channels, out_channels, 3, padding=1, stride=2, output_padding=1),
            act_fn,
            nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),
            act_fn,
            nn.ConvTranspose2d(out_channels, in_channels, 3, padding=1),
        )

    def forward(self, x):
        output = self.linear(x)
        output = output.view(-1, 8 * self.out_channels, 14, 14)
        output = self.conv(output)
        return output


class Autoencoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device):
        """
        Autoencoder class

        :param encoder: Encoder class
        :param decoder: Decoder class
        :param device: Pytorch device
        """
        super().__init__()

        self.device = device

        self.encoder = encoder
        self.encoder.to(device)

        self.decoder = decoder
        self.decoder.to(device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ConvolutionalAutoencoder:
    def __init__(self, autoencoder: Autoencoder):
        """
        Main model of autoencoder for face comparison

        :param autoencoder: Autoencoder class
        """
        self.network = autoencoder
        self.optimizer = torch.optim.Adam(self.network.parametrs(), lr=1e-3)

    def train(self, loss_function, epochs, batch_size, training_set, validation_set):
        """
        Function for training neural network (autoencoder)

        :param loss_function: Loss function
        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param training_set: Set of images for training
        :param validation_set: Set of images for validation
        :return: dict with losses
        """
        log_dict = {
            'training_loss_per_batch': [],
            'validation_loss_per_batch': []
        }

        def init_weights(module: nn.Module):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

        self.network.apply(init_weights)

        train_loader = DataLoader(training_set, batch_size)
        val_loader = DataLoader(validation_set, batch_size)

        self.network.train()
        self.network.to(self.network.device)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            loss = 0
            val_loss = 0

            print("Training...")

            for image in tqdm(train_loader):
                self.optimizer.zero_grad()
                images = image.to(self.network.device)
                output = self.network(images)
                loss = loss_function(output, images.view(-1, 3, 112, 112))
                loss.backward()
                self.optimizer.step()

                log_dict['training_loss_per_batch'].append(loss.item())

            print("Validation...")

            for val_image in tqdm(val_loader):
                with torch.no_grad():
                    val_image = val_image.to(self.network.device)
                    output = self.network(val_image)
                    val_loss = loss_function(output, val_image.view(-1, 3, 112, 112))

                    log_dict['validation_loss_per_batch'].append(val_loss.item())

            print()
            print(f"training_loss: {round(loss.item(), 4)}\nvalidation_loss: {round(val_loss.item(), 4)}")

        return log_dict

    def auto_encode(self, x):
        return self.network(x)

    def encode(self, x):
        encoder = self.network.encoder
        return encoder(x)

    def decode(self, x):
        decoder = self.network.decoder
        return decoder(x)

    def comparison_similarity(self, img1, img2):
        """
        Function for getting visual similarity between two photos

        :param img1: cv2 image
        :param img2: cv2 image
        :return: percents of similarity
        """
        try:
            image1 = cv2.resize(img1, (112, 112))
            image1 = np.array(image1, "uint8")

            image2 = cv2.resize(img2, (112, 112))
            image2 = np.array(image2, "uint8")

            my_transform = transforms.Compose([
                transforms.ToTensor,
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            image1 = my_transform(image1)
            image2 = my_transform(image2)

            image1 = image1.to(self.network.device)
            image2 = image2.to(self.network.device)

            with torch.no_grad():
                image1_encodings = self.encode(image1)
                image2_encodings = self.encode(image2)

            similarity_score = f.cosine_similarity(image1_encodings, image2_encodings)
            similarity_score = similarity_score.cpu().detach().item()
            similarity_score = round(similarity_score, 3)

            return similarity_score * 100
        except:
            return FileNotFoundError

    def visual_similarity(self, img, dataset, features):
        """
        Function for getting similarity with dataset faces

        :param img: cv2 image
        :param dataset: dataset with images for comparison
        :param features: array with encoded images
        :return: array with most similar faces, percents and name
        """
        image = cv2.resize(img, (112, 112))
        image = np.array(image, "uint8")

        my_transform = transforms.Compose([
            transforms.ToTensor,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        image = my_transform(image)
        image = image.to(self.network.device)

        with torch.no_grad():
            image_encodings = self.encode(image)

        similarity_scores = [f.cosine_similarity(image_encodings, x) for x in features]
        similarity_scores = [x.cpu().detach().item() for x in similarity_scores]
        similarity_scores = [round(x, 3) for x in similarity_scores]

        scores = pd.Series(similarity_scores)
        scores = scores.sort_values(ascending=False)

        most_similar = []

        for i in range(len(dataset.images)):
            cont = False
            idx = scores.index[i]
            if len(most_similar) == 5:
                break
            for similar in most_similar:
                if scores[idx] * 100 == similar[1]:
                    cont = True
                    break
            if not cont:
                most_similar.append([
                    dataset.images[idx],
                    scores[idx] * 100,
                    dataset.names[dataset[idx][1]]
                ])

        return most_similar

    def save_model(self, parameters_path="autoencoder_weights.pth"):
        torch.save(self.network.state_dict(), parameters_path)
        print(f"Model saved {parameters_path}")

    def load_model(self, parameters_path="autoencoder_weights.pth"):
        try:
            if torch.cuda.is_available():
                self.network.load_state_dict(torch.load(parameters_path))
            else:
                self.network.load_state_dict(torch.load(parameters_path, map_location=torch.device("cpu")))
        except:
            return FileNotFoundError
