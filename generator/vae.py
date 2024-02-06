import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.distributions
import torch.utils
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy


class Encoder(nn.Module):
    def __init__(self, input_size, latent_dims, depth=3):
        super(Encoder, self).__init__()
        layers = list()
        assert latent_dims < input_size
        step = (input_size - latent_dims) // depth if depth > 0 else 0
        for i in range(1, depth):
            layers.append(nn.Linear(input_size - step *
                          (i-1), input_size - step*(i)))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(input_size - step*(i)))
            layers.append(nn.Dropout(p=0.1))
        layers.append(nn.Linear(input_size - step*(depth-1), latent_dims))
        self.fc_mu = nn.Sequential(*copy.deepcopy(layers))
        self.fc_logvar = nn.Sequential(*layers)

    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dims, output_size, depth=3):
        super(Decoder, self).__init__()
        layers = list()
        assert latent_dims < output_size

        step = (output_size - latent_dims)//depth if depth > 0 else 0

        latent_dims += 4  # 1 team, 2 ball pos, 1 ball control

        for i in range(1, depth):
            layers.append(nn.Linear(latent_dims + step *
                          (i-1), latent_dims + step*(i)))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(latent_dims + step*(i)))
            layers.append(nn.Dropout(p=0.1))
        layers.append(nn.Linear(latent_dims + step*(depth-1), output_size))
        layers.append(nn.Sigmoid())

        self.linear = nn.Sequential(*layers)

    def forward(self, z, team, ballPosition, ballControl):
        z = torch.cat(
            (z, team.unsqueeze(1), ballPosition, ballControl.unsqueeze(1)), dim=-1)
        return self.linear(z)


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size, latent_dims, depth=3):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(input_size, latent_dims, depth)
        self.decoder = Decoder(latent_dims, input_size, depth)

    def forward(self, x, team, ballPosition, ballControl):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        print(std.dtype)
        z = torch.randn_like(std) * std + mu

        return self.decoder(z, team, ballPosition, ballControl), mu, logvar
