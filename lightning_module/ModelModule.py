import pytorch_lightning as pl
import torch.nn as nn
import torch
import logging
import os



class LigtningModel(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr
        
        self.loss = nn.MSELoss()
        
        # Initialize an empty tensor to store the concatenated data
        self.prediction = None
        self.target = None
        

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    #def on_training_epoch_end(self, outputs):
        #self.logger.info(f"Epoch {self.current_epoch} training_loss: {ouputs.item()}")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return loss, y_hat, y
        
    #def on_validation_epoch_end(self, outputs):
        #self.logger.info(f"Epoch {self.current_epoch} val_loss: {ouputs[0].item()}")
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        
        self.log('test_loss', loss)
        
        if self.prediction is None:
            self.prediction = y_hat
            self.target = y
        else:
            # Stack each batch along the batch dimension (axis 0)
            self.prediction = torch.cat((self.prediction, y_hat), dim=0)
            self.target = torch.cat((self.target, y), dim=0)
    """
    def validation_epoch_end(self, outputs):
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
    """
    
    def return_prediction(self):
        return self.prediction.cpu().numpy(), self.target.cpu().numpy()

    #def on_test_epoch_end(self):
        """
        trading = TradingModel(self.time_length, self.step_size, self.prediction, self.target, self.scaler)
        
        self.loss_pred, self.cost_x_generation_mpc, self.prediction_store, self.trading_plan = trading.generate_trading_plan()
        self.consistency_loss_pred, self.consistency_loss_vol, self.consistency_loss_pred_abs, self.consistency_loss_vol_abs = ConsistencyMetrics.calculate(self.prediction_store,  self.trading_plan, self.step_size, self.time_length)
        """
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr )

