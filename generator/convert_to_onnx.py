import torch
from vae import VariationalAutoencoder
from dataset import MyDataset
from torch.utils.data import DataLoader


latent_dims = 10
granularity = 0.01
autoencoder = VariationalAutoencoder(int(1./granularity)**2, latent_dims, 1)
autoencoder.load_state_dict(torch.load("model.pt"))
autoencoder.eval()


dataset = MyDataset(csv_file="merged.csv",
                    distributionGranurality=granularity)

dataloader = DataLoader(dataset, batch_size=1)
sample = next(iter(dataloader))
x = sample["distribution"]
x = x.view(-1, x.size(1)**2)
t = sample["team"]
bp = sample["ballPosition"]
bc = sample["ballControl"]

torch.onnx.export(autoencoder, (x, t, bp, bc), "model.onnx", verbose=True)
