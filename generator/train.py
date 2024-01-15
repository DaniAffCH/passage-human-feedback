import torch
from vae import VariationalAutoencoder

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        # clean NaN
        self.data = self.clean(self.data)

    def clean(self, df):

        min_value_x = df.filter(like='x_pos_').min().min()
        max_value_x = df.filter(like='x_pos_').max().max()

        min_value_y = df.filter(like='y_pos_').min().min()
        max_value_y = df.filter(like='y_pos_').max().max()

        for column in df.columns:
            mask = df[column].isnull()
            if column.startswith('x_pos_'):
                df.loc[mask, column] = np.random.uniform(
                    min_value_x, max_value_x, size=np.sum(mask))
            elif column.startswith('y_pos_'):
                df.loc[mask, column] = np.random.uniform(
                    min_value_y, max_value_y, size=np.sum(mask))

        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        values = list()

        for i in range(1, 8):
            x_col = f'x_pos_{i}'
            y_col = f'y_pos_{i}'

            values.append(self.data.loc[idx, x_col])
            values.append(self.data.loc[idx, y_col])

        return {'positions': torch.tensor(values, dtype=torch.float32), 'team': torch.tensor(self.data.loc[idx, "team"], dtype=torch.long)}


def train(autoencoder, data, epochs=20000):
    opt = torch.optim.Adam(autoencoder.parameters(), 1e-3)
    for epoch in range(epochs):
        losses = list()
        for d in data:

            x = d["positions"].to(device)
            t = d["team"].to(device)
            opt.zero_grad()

            x_hat = autoencoder(x, t)
            print(x_hat)
            loss = ((x - x_hat)**2).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        print("Epoch:", epoch, "avg loss:", sum(losses)/len(losses))
    return autoencoder


if __name__ == "__main__":
    latent_dims = 7
    autoencoder = VariationalAutoencoder(14, latent_dims).to(device)

    # Load your CSV file into a PyTorch dataset
    dataset = MyDataset("processed.csv")

    # Split the dataset into train and test sets
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.01, random_state=42)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    autoencoder = train(autoencoder, train_dataloader)
