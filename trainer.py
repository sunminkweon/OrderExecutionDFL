import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from utils import logger
import torch

class Trainer:
    def __init__(self, model, args, ticker):
        self.model = model
        self.args = args
        self.ticker =ticker
        self.trainer = None

    def initialize_callbacks_and_logger(self, model_name):
        self.tqdm_cb, self.tb_logger = logger.initialize_callbacks_and_logger(model_name )

    def initialize_model_callback(self, dirpath):
        logger_instance = pl.callbacks.ModelCheckpoint(
            dirpath=dirpath,
            filename='{epoch}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1
        )
        return logger_instance

    def train_model(self, accelerator, train_loader, val_loader, max_epochs, dirpath=None):
        self.initialize_callbacks_and_logger(self.model.__class__.__name__ +"/"+self.model.model.__class__.__name__+"/"+self.ticker)
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor='train_task_loss_epoch',
            min_delta=0.0001,
            patience=5,  # Adjust patience as needed
            mode='min'
        )
        # Initialize logger only if dirpath exists
        if dirpath is not None:
            logger_instance = self.initialize_model_callback(dirpath)
            callbacks = [self.tqdm_cb, logger_instance, early_stopping_callback]
        else:
            callbacks = [self.tqdm_cb, early_stopping_callback]
        self.trainer = pl.Trainer(
            accelerator=accelerator,
            max_epochs=max_epochs,
            callbacks=callbacks,
            logger=self.tb_logger,
            log_every_n_steps=1
        )
        
        self.trainer.fit(self.model, train_loader, val_loader)

        return self.trainer.callback_metrics["val_loss"].item()

    def test_model(self, test_loader) :
        return self.trainer.test(self.model, test_loader)
    
    
    def test_model_tmp(self, test_loader) :
        self.initialize_callbacks_and_logger(self.model.__class__.__name__ +"/"+self.model.model.__class__.__name__+"/"+self.ticker)
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=5,  # Adjust patience as needed
            mode='min'
        )
        callbacks = [self.tqdm_cb, early_stopping_callback]
        self.trainer = pl.Trainer(
            max_epochs=1,
            callbacks=callbacks,
            logger=self.tb_logger,
            log_every_n_steps=1
        )
        print("ASdf")
        return self.trainer.test(self.model, test_loader)