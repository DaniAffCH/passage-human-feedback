import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utlis_ import Distribution

field_start_x = 0
field_end_x = 690
field_start_y = 0
field_end_y = 464


class MyDataset(Dataset):
    def __init__(self, csv_file, distributionGranurality=0.01):
        self.data = pd.read_csv(csv_file)
        self.data = self.scale(self.data)
        self.distributionGranurality = distributionGranurality

    def scale(self, df):

        for column in df.columns:
            if column.startswith('x_pos_'):
                df[column] = (df[column] - field_start_x)/field_end_x
            elif column.startswith('y_pos_'):
                df[column] = (df[column] - field_start_y)/field_end_y

        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = Distribution(self.distributionGranurality)
        d.generateDistribution(self.data.loc[idx])
        return {"distribution": torch.tensor(d.matrix, dtype=torch.float32),
                "team": torch.tensor(self.data.loc[idx, "team"], dtype=torch.long),
                "ballPosition": torch.tensor([self.data.loc[idx, "x_pos_ball"], self.data.loc[idx, "y_pos_ball"]], dtype=torch.float32),
                "ballControl": torch.tensor(self.data.loc[idx, "ballControl"])}
