import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.distributions
import torch.utils
import torch.nn.functional as F
import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, input_size, latent_dims, depth=3):
        super(Encoder, self).__init__()
        layers = list()
        assert latent_dims < input_size
        step = (input_size - latent_dims)//depth if depth > 0 else 0
        for i in range(1, depth):
            layers.append(nn.Linear(input_size - step *
                          (i-1), input_size - step*(i)))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(input_size - step*(i)))
            layers.append(nn.Dropout(p=0.1))
        layers.append(nn.Linear(input_size - step*(depth-1), latent_dims))
        layers.append(nn.Sigmoid())

        self.linear = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear(x)


class Decoder(nn.Module):
    def __init__(self, latent_dims, output_size, depth=3):
        super(Decoder, self).__init__()
        layers = list()
        assert latent_dims < output_size

        step = (output_size - latent_dims)//depth if depth > 0 else 0

        latent_dims += 1

        for i in range(1, depth):
            layers.append(nn.Linear(latent_dims + step *
                          (i-1), latent_dims + step*(i)))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(latent_dims + step*(i)))
            layers.append(nn.Dropout(p=0.1))
        layers.append(nn.Linear(latent_dims + step*(depth-1), output_size))
        layers.append(nn.Sigmoid())

        self.linear = nn.Sequential(*layers)

    def forward(self, z, team):
        z = torch.cat((team.unsqueeze(1), z), dim=-1)
        return self.linear(z)


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size, latent_dims, depth=3) -> None:
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(input_size, latent_dims, depth)
        self.decoder = Decoder(latent_dims, input_size, depth)

    def forward(self, x, team):
        z = self.encoder(x)
        print(z.shape, team.shape)
        return self.decoder(z, team)
