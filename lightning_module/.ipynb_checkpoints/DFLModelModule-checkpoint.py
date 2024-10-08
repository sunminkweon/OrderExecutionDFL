import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
from optimizer.End2EndTradingModel import E2ETradingModel



class LightningEndtoEndModel(pl.LightningModule):
    def __init__(self, model, decision_layer, scaler, time_length, step, learning_rate):
        super().__init__()
        self.time_length = time_length
        self.model = model
        self.decision_layer = decision_layer
        self.step = step
        self.prediction  = None
        self.target = None
        self.learning_rate = learning_rate

        self.scaler = scaler
        self.scaler_mean = torch.tensor(self.scaler.mean_).to('cuda')
        self.scaler_std = torch.sqrt (torch.tensor(self.scaler.var_)).to('cuda')
        self.loss_pred = nn.MSELoss()

        self.loss_pred_weight = 1
        self.loss_decision_weight = 1
         # Initialize weights to 1 for diagonal elements and 0 for others


    def training_step(self, batch, batch_idx):
        x, y = batch
        
        #loss for prediciton
        pred = self.model(x)        
        loss_pred = self.loss_pred(pred, y)

        # for decision part
        pred_decision = torch.exp(self.scaler_std * pred + self.scaler_mean)
        y_decision = torch.exp(self.scaler_std * y + self.scaler_mean)
        trading = DecisionAwareTradingModel(y_decision)
        
        pred_ = torch.sqrt(1/pred_decision).clone()
        for t in range(self.time_length) :
            left_vol = trading.return_left_vol_input(t)
            # for constraint
            time_constraint = torch.zeros( (self.time_length), device = 'cuda', dtype=torch.float64)
            time_constraint[self.time_length-t:] = 1

            decision = self.decision_layer(pred_[:,t], left_vol, time_constraint)
            #prediction for store
            trading.generate_trading_plan(t, decision[0], pred_decision[:,t])
    
        cost_x_generation_mpc = trading.return_cost_x()
        loss_decision = torch.mean(torch.sum(cost_x_generation_mpc, axis=1), axis=0)

        loss = loss_decision
        #loss = loss_pred * self.loss_pred_weight + self.loss_decision_weight * loss_decision
        self.log('train_loss', loss)
        return loss

    #def on_training_epoch_end(self, outputs):
        #self.logger.info(f"Epoch {self.current_epoch} training_loss: {ouputs.item()}")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        #loss for prediciton
        pred = self.model(x)        
        loss_pred = self.loss_pred(pred, y)

        # for decision part
        pred_decision = torch.exp(self.scaler_std * pred + self.scaler_mean)
        y_decision = torch.exp(self.scaler_std * y + self.scaler_mean)
        trading = DecisionAwareTradingModel(y_decision)
        
        pred_ = torch.sqrt(1/pred_decision).clone()
        for t in range(self.time_length) :
            left_vol = trading.return_left_vol_input(t)
            # for constraint
            time_constraint = torch.zeros( (self.time_length), device = 'cuda', dtype=torch.float64)
            time_constraint[self.time_length-t:] = 1

            decision = self.decision_layer(pred_[:,t], left_vol, time_constraint)
            #prediction for store
            trading.generate_trading_plan(t, decision[0], pred_decision[:,t])
    
        cost_x_generation_mpc = trading.return_cost_x()
        loss_decision = torch.mean(torch.sum(cost_x_generation_mpc, axis=1), axis=0)

        loss = loss_decision
        #loss = loss_pred * self.loss_pred_weight + self.loss_decision_weight * loss_decision
        self.log('val_loss', loss)
        return loss
        
    #def on_validation_epoch_end(self, outputs):
        #self.logger.info(f"Epoch {self.current_epoch} val_loss: {ouputs[0].item()}")
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        #loss for prediciton
        pred = self.model(x)        
        loss_pred = self.loss_pred(pred, y)

        # for decision part
        pred_decision = torch.exp(self.scaler_std * pred + self.scaler_mean)
        y_decision = torch.exp(self.scaler_std * y + self.scaler_mean)
        trading = DecisionAwareTradingModel(y_decision)
        
        pred_ = torch.sqrt(1/pred_decision).clone()
        for t in range(self.time_length) :
            left_vol = trading.return_left_vol_input(t)
            # for constraint
            time_constraint = torch.zeros( (self.time_length), device = 'cuda', dtype=torch.float64)
            time_constraint[self.time_length-t:] = 1

            decision = self.decision_layer(pred_[:,t], left_vol, time_constraint)
            #prediction for store
            trading.generate_trading_plan(t, decision[0], pred_decision[:,t])
    
        cost_x_generation_mpc = trading.return_cost_x()
        loss_decision = torch.mean(torch.sum(cost_x_generation_mpc, axis=1), axis=0)

        loss = loss_decision
        #loss = loss_pred * self.loss_pred_weight + self.loss_decision_weight * loss_decision

        cost_x_generation_mpc = trading.return_cost_x()
        pred_store = trading.return_prediction()
        
        loss = torch.mean(torch.sum(cost_x_generation_mpc, axis=1), axis=0)
        self.log('test_loss', loss)
        if self.prediction is None:
            self.prediction = pred_store.contiguous()
            self.target = pred_store.contiguous()
        else:
            # Stack each batch along the batch dimension (axis 0)
            self.prediction = torch.cat((self.prediction, pred_store.contiguous()), dim=0).contiguous()
            self.target = torch.cat((self.target, pred_store.contiguous()), dim=0).contiguous()
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
        return self.prediction, self.target

    #def on_test_epoch_end(self):
        """
        trading = TradingModel(self.time_length, self.step_size, self.prediction, self.target, self.scaler)
        
        self.loss_pred, self.cost_x_generation_mpc, self.prediction_store, self.trading_plan = trading.generate_trading_plan()
        self.consistency_loss_pred, self.consistency_loss_vol, self.consistency_loss_pred_abs, self.consistency_loss_vol_abs = ConsistencyMetrics.calculate(self.prediction_store,  self.trading_plan, self.step_size, self.time_length)
        """
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01 )
    

class DecisionAwareTradingModel:
    """
    TradingModel class for generating trading plans based on predictions.

    Args:
        time_length (int): Length of the time sequence.
        step_size (list): List of step sizes for trading, in other words, the period of updating prediction .
        prediction (np.array): Array containing predictions, size : ( N, time_length(prediction_length), feature)
        target (np.array):: Array containing real data, size : ( N, time_length(prediction_length), feature)
    """

    def __init__(self, target):
        self.time_length = target.shape[-1]
        self.target = target.clone()
        #self.target no slid
        self.target_no_sw = torch.zeros((self.target.shape[0], self.time_length), device=torch.device('cuda'))
        for t in range(self.time_length) : 
            self.target_no_sw = self.target[:,0]
    
        self.prediction_store = torch.zeros((self.target.shape[0],  self.time_length,  self.time_length), device=torch.device('cuda'), dtype=torch.float64)
        self.trading_plan = torch.zeros((self.target.shape[0],  self.time_length,  self.time_length), device=torch.device('cuda'), dtype=torch.float64)
        self.cost_x_generation_mpc = torch.zeros((self.target.shape[0],  self.time_length), device=torch.device('cuda'), dtype=torch.float64)
        self.left_vol = torch.zeros(( self.target.shape[0],  self.time_length+1), device=torch.device('cuda'), dtype=torch.float64)
        self.left_vol[:,0] = 10000

    def generate_trading_plan(self, t, decision, pred):
        """
        Generate trading plans based on predictions.

        Returns:
            tuple: A tuple containing loss_pred, trading_cost, prediction_store, and trading_plan.
        """
        torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

        self.prediction = pred
        self.decision = decision[:,:self.time_length-t].clone()
        self.update_prediction_store(t)
        self.update_trading_plan(t)
        self.update_left_vol(t)
        self.update_cost_x_generation_mpc(t)
        
    def update_prediction_store(self, t):
        """
        Update the prediction store.
    
        Args:
            t (int): Current time index.
        """
        self.prediction_store[:, t, t:] = self.prediction[:, :self.time_length- t].clone()

    
    def update_trading_plan(self, t):
        self.trading_plan[:, t, t:] = self.decision.clone()
            
            
    def update_left_vol(self, t):
        self.left_vol[:, t + 1] = self.left_vol[:, t] - self.trading_plan[:, t, t]
    
    def update_cost_x_generation_mpc(self, t):
        """
        Update the cost x generation MPC.

        Args:
            t (int): Current time index.
        """
        trading_plan_square = torch.pow(self.trading_plan[:, t, t].clone(), 2)
        target_plus_epsilon = (self.target_no_sw[:, t].clone())
        division_result = trading_plan_square / target_plus_epsilon
        self.cost_x_generation_mpc[:, t] = division_result

    def return_left_vol_input(self,t):
        return self.left_vol[:,t]

    def return_cost_x(self):
        return self.cost_x_generation_mpc

    def return_prediction(self):
        return self.prediction_store