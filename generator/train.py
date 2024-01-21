import torch
from vae import VariationalAutoencoder

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import numpy as np
from dataset import MyDataset
from torch.utils.data import DataLoader


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(autoencoder, train_data, test_data, epochs=1000):
    opt = torch.optim.Adam(autoencoder.parameters(), 1e-4)
    for epoch in range(epochs):
        losses = list()
        val_losses = list()
        autoencoder.train()
        for d in train_data:

            x = d["distribution"].to(device)
            x = x.view(-1, x.size(1)**2)
            t = d["team"].to(device)
            bp = d["ballPosition"].to(device)
            bc = d["ballControl"].to(device)

            opt.zero_grad()

            x_hat, mu, logvar = autoencoder(x, t, bp, bc)
            reconstruction_loss = torch.nn.functional.mse_loss(
                x_hat, x, reduction='sum')

            kl_divergence_loss = -0.5 * \
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            total_loss = reconstruction_loss + kl_divergence_loss/len(d)

            total_loss.backward()
            opt.step()
            losses.append(total_loss.item())

        autoencoder.eval()
        with torch.no_grad():
            for d in test_data:
                x = d["distribution"].to(device)
                x = x.view(-1, x.size(1)**2)
                t = d["team"].to(device)
                bp = d["ballPosition"].to(device)
                bc = d["ballControl"].to(device)

                x_hat, mu, logvar = autoencoder(x, t, bp, bc)

                reconstruction_loss = torch.nn.functional.mse_loss(
                    x_hat, x, reduction='sum')

                kl_divergence_loss = -0.5 * \
                    torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                total_loss = reconstruction_loss + kl_divergence_loss

                val_losses.append(total_loss.item())

        print("Epoch:", epoch, "train loss:", np.array(
            losses).mean(), "val_losses", np.array(val_losses).mean())
    torch.save(autoencoder.state_dict(), "model.pt")


if __name__ == "__main__":
    latent_dims = 10
    granularity = 0.01
    batch_size = 8
    autoencoder = VariationalAutoencoder(
        int(1./granularity)**2, latent_dims, 1).to(device)

    dataset = MyDataset(csv_file="merged.csv",
                        distributionGranurality=granularity)

    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.2, random_state=42)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    autoencoder = train(autoencoder, train_dataloader, test_dataloader)
