
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import Callback
import os



def initialize_callbacks_and_logger(model_name):
    """
    Initialize Lightning callbacks and logger.
    """
    tqdm_cb = TQDMProgressBar(refresh_rate=1)
    tb_logger = TensorBoardLogger(save_dir='./log',name=model_name)
    return tqdm_cb, tb_logger

def initialize_model_callback(dir):
    ckpt_cb = ModelCheckpoint(
    save_top_k= 1,
    monitor="val_loss",
    mode="min",
    dirpath=dir,
    filename="{epoch:02d}-{val_loss:.2f}",
    )
    return ckpt_cb