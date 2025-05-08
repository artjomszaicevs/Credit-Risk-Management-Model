import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SimpleTensorDataset(Dataset):
    def __init__(self, data_dict):
        self.data = list(data_dict.items())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id_, X = self.data[idx]
        return torch.tensor(X, dtype=torch.float32), id_


class SimpleAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(B, T * D)
        z = self.encoder(x_flat)
        x_hat = self.decoder(z).view(B, T, D)
        return z, x_hat