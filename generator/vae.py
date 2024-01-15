import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.distributions
import torch.utils
import torch.nn.functional as F
import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, (input_size+latent_dims)//2)
        self.linear2 = nn.Linear((input_size+latent_dims)//2, latent_dims)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class Decoder(nn.Module):
    def __init__(self, latent_dims, output_size):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims+1, (latent_dims+1+output_size)//2)
        self.linear2 = nn.Linear((latent_dims+1+output_size)//2, output_size)

    def forward(self, z, team):
        z = torch.cat((team.unsqueeze(1), z), dim=-1)
        z = F.relu(self.linear1(z))
        return F.relu(self.linear2(z))


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size, latent_dims) -> None:
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(input_size, latent_dims)
        self.decoder = Decoder(latent_dims, input_size)

    def forward(self, x, team):
        z = self.encoder(x)
        return self.decoder(z, team)
