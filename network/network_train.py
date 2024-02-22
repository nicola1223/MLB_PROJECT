import torch
from torch import nn
from datasets import TrainDataset, divide_dataset
from network import Encoder, Decoder, Autoencoder, ConvolutionalAutoencoder

device = torch.device("cuda:0")
model = ConvolutionalAutoencoder(Autoencoder(Encoder(), Decoder(), device))

dataset = TrainDataset("./network/Dataset")
training_data, validation_data = divide_dataset(dataset)

log_dict = model.train(
    nn.MSELoss(),
    epochs=20,
    batch_size=10,
    training_set=training_data,
    validation_set=validation_data
)

model.save_model()
