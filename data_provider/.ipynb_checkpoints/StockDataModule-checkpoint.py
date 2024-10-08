import os
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from data_provider import data_reader, data_preprocessor, data_splitter
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing 

class StockDataModule(pl.LightningDataModule):
    def __init__(self,
                 directory: str = "./data/stock_data_prepro",
                 ticker: str = 'AAPL_5min',
                 time_length: int = 78,
                 prediction_length : int= 78,
                 context_length : int = 2*78,
                 wanted_interval : int = 5,
                 train_ratio: float = 0.6,
                 val_ratio: float = 0.2,
                 batch_size: int = 64,
                 ):
        super().__init__()
        self.directory = directory
        self.ticker = ticker
        self.time_length = time_length
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.wanted_interval = wanted_interval
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.scaler = sklearn.preprocessing.StandardScaler() #Scaler()

    def setup(self, stage=None):
        # Load and preprocess your data
        df = data_reader.read_stock_data(self.directory, self.ticker)
        
        # Log transformation
        df_vol = df['volume']
        df_vol, self.time_length = data_preprocessor.preprocess_interval(df_vol , self.time_length, self.wanted_interval)

        # split data into training, val, testing and time length becausse split data with day ratio
        train_df, validation_df, test_df = data_splitter.split_data(df_vol, self.time_length, self.train_ratio, self.val_ratio)

        train_np = np.array(train_df).reshape(-1,1)
        validation_np = np.array( validation_df).reshape(-1,1)
        test_np = np.array(test_df).reshape(-1,1)

        train_df = self.scaler.fit_transform(np.array(train_df).reshape(-1,1))
        validation_df = self.scaler.transform(np.array(validation_df).reshape(-1,1))
        test_df = self.scaler.transform(np.array(test_df).reshape(-1,1))

        # Preprocess data into input-output pairs
        train_X, train_Y = data_preprocessor.preprocess_sliding_window(train_df, self.time_length, self.context_length, self.prediction_length)
        val_X, val_Y = data_preprocessor.preprocess_sliding_window(validation_df, self.time_length, self.context_length, self.prediction_length)
        test_X, test_Y = data_preprocessor.preprocess_sliding_window(test_df, self.time_length, self.context_length, self.prediction_length)
        
        # Convert to PyTorch tensors
        self.train_dataset = TensorDataset(torch.Tensor(train_X), torch.Tensor(train_Y))
        self.val_dataset = TensorDataset(torch.Tensor(val_X), torch.Tensor(val_Y))
        self.test_dataset = TensorDataset(torch.Tensor(test_X), torch.Tensor(test_Y))

        return self.scaler

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=16, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=16)

def initialize_data_module(directory, ticker, time_length, prediction_length, context_length,  wanted_interval, train_ratio, val_ratio, batch_size):
    
    data_module = StockDataModule(directory=directory, ticker=ticker, prediction_length= prediction_length, time_length=time_length, context_length=context_length, wanted_interval = wanted_interval, train_ratio=train_ratio, val_ratio=val_ratio, batch_size=batch_size)
    
    scaler = data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    return scaler, train_loader, val_loader, test_loader