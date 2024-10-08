import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
    
class LightningDFLModel(pl.LightningModule):
    def __init__(self, model, adjustment_layer, scaler, time_length,  learning_rate, weight, train_loader, val_loader, test_loader) :
        super().__init__()
        self.time_length = time_length
        self.model = model
        self.adjustment_layer = adjustment_layer
        self.prediction  = None
        self.target = None
        self.learning_rate = learning_rate

        self.scaler = scaler 
        self.scaler_mean = torch.tensor(self.scaler.mean_).to('cuda')
        self.scaler_std = torch.sqrt (torch.tensor(self.scaler.var_)).to('cuda')
        
        self.loss_pred_weight = weight[1]
        self.loss_task_weight = weight[0]
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
    def forward(self, x, left_vol, prev, t):
        # concanate the previous realized trading volume as input
        if t != 0:
            padding = torch.zeros( (x.shape[0], t), device=torch.device('cuda'))   
            
            x = torch.cat(( prev , x[:,:self.time_length - t] ), axis=1)    
            #x = torch.cat(( x[:,:self.time_length - t], padding ), axis=1)    
        for layer in self.adjustment_layer:
            x = layer(x)
        return x[:,t:]
           

    def calc_loss_prediction(self , y_hat, y, t) :
        prediction_loss = ( (torch.log(y_hat[:,:]) - torch.log(y[:,0,t:])) )**2
        return torch.mean (prediction_loss)
    
    def on_train_start(self):
        torch.set_grad_enabled(False)
        # Calculate loss for training set
        loss_task = torch.zeros(1, device='cuda')
        for batch_idx, batch in enumerate(self.train_loader):
            batch = [item.to('cuda') for item in batch]
            loss_task +=self.validation_step(batch, batch_idx)
        
        self.log('train_task_loss_start', loss_task/ (batch_idx+1))
       
        loss_task = 0
        for batch_idx, batch in enumerate(self.val_loader):
            batch = [item.to('cuda') for item in batch]
            loss_task +=self.validation_step(batch, batch_idx)
            
        self.log('val_task_loss_start', loss_task/ (batch_idx+1))
        
        loss_task = 0
        for batch_idx, batch in enumerate(self.test_loader):
            batch = [item.to('cuda') for item in batch]
            loss_task +=self.test_step(batch, batch_idx)

        self.log('test_task_loss_start', loss_task/ (batch_idx+1))  
  
    def calc_loss_task(self, x, y, test=False):
        
        loss_pred = 0 
        target = y[:,0].clone()
        y = torch.exp( y*self.scaler_std + self.scaler_mean)
        
        trading = DecisionFocusedTradingModel(y, self.scaler)
        
        left_vol = trading.return_left_vol_input(0)

        y_hat =  torch.exp( self(x[:,0], left_vol, target[:,0], 0)*self.scaler_std + self.scaler_mean)
        
        trading.generate_trading_plan(0, y_hat)
        
        loss_pred +=  self.calc_loss_prediction(y_hat, y, 0 )
        for t in range(1,self.time_length,1) :
            left_vol = trading.return_left_vol_input(t)   
            y_hat =  torch.exp( self(x[:,t], left_vol, target[:, :t], t)*self.scaler_std + self.scaler_mean) 
            trading.generate_trading_plan(t, y_hat)

            loss_pred += self.calc_loss_prediction(y_hat, y, t)
    
        cost_x_generation_mpc = trading.return_cost_x()
        
        loss_task = torch.mean(torch.sum(cost_x_generation_mpc, axis=1), axis=0)
        
        if test : 
            pred_store = trading.return_prediction()
            return loss_task, loss_pred, pred_store
        else :
            return loss_task, loss_pred
    
    def calc_r2_score(prediction, y)
        r2_score = 0
        
        y = torch.exp( y*self.scaler_std + self.scaler_mean)
        target = y[:,0].clone()
        target_mean = torch.mean(target, axis=0)
        
        tss = 0
        rss = 0
        for t in range(self.time_length) :
            tss +=  np.sum((np.log(target[:,t:] -target_mean[t:])**2)
            rss +=  np.sum((np.log(prediction)[:,t,t:] -np.log(target)[:,t:])**2)

        r2_score = 1-(rss/tss) 
        return r2_score
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        pred = self.model(x)        
        loss_task, loss_pred = self.calc_loss_task(pred, y)
        
        self.log('train_task_loss', loss_task, on_epoch=True)
        loss = loss_pred * self.loss_pred_weight + self.loss_task_weight * loss_task
        self.log('train_pred_loss', loss_pred, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        pred = self.model(x)        
        loss_task, loss_pred = self.calc_loss_task(pred, y)
        self.log('val_loss', loss_task, on_epoch=True)
        return loss_task
        
    #def on_validation_epoch_end(self, outputs):
        #self.logger.info(f"Epoch {self.current_epoch} val_loss: {ouputs[0].item()}")
        
    def test_step(self, batch, batch_idx):
        x, y = batch
     
        pred = self.model(x)        
        loss_task, loss_pred , pred_store = self.calc_loss_task(pred, y, True)
        r2_score = self.calc_r2_score(pred, y)

        self.log('test_loss',loss_task, on_epoch=True)
        self.log('r2_score',r2_score, on_epoch=True)
        if batch_idx == 0 :
            self.prediction = pred_store.contiguous()
            self.target = pred_store.contiguous()
        else:
            # Stack each batch along the batch dimension (axis 0)
            self.prediction = torch.cat((self.prediction, pred_store.contiguous()), dim=0).contiguous()
            self.target = torch.cat((self.target, pred_store.contiguous()), dim=0).contiguous()
        return loss_task
                    

    def return_prediction(self):
        return self.prediction, self.target

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr= self.learning_rate )
    
 
    
    
class DecisionFocusedTradingModel:
    """
    TradingModel class for generating trading plans based on predictions.

    Args:
        time_length (int): Length of the time sequence.
        prediction (np.array): Array containing predictions, size : ( N, time_length(prediction_length), time_length(prediction_length))
        target (np.array):: Array containing real data, size : ( N, time_length(prediction_length), time_length(prediction_length))
    """

    def __init__(self, target, scaler):
        """
        Initialize the DecisionFocusedTradingModel with the target data and a scaler for normalizing.

        Args:
            target (torch.Tensor): The real target data used for trading and comparison with predictions.
            scaler (object): Scaler object that contains the mean and variance to normalize data.
        """
        self.time_length = target.shape[-1]
        self.target = target.clone()
        
        self.scaler = scaler 
        self.mean = torch.tensor(self.scaler.mean_).to('cuda')
        self.std = torch.sqrt (torch.tensor(self.scaler.var_)).to('cuda')
        
        # Create a tensor to hold the target without any sliding window, initialized to zeros
        self.target_no_sw = torch.zeros((self.target.shape[0], self.time_length), device=torch.device('cuda'))
        
        # We need only one-day target for calculation
        self.target_no_sw = self.target[:,0] 
    
        self.prediction_store = torch.zeros((self.target.shape[0],  self.time_length,  self.time_length), device=torch.device('cuda'))
        self.trading_plan = torch.zeros((self.target.shape[0],  self.time_length,  self.time_length), device=torch.device('cuda'))
        self.cost_x_generation_mpc = torch.zeros((self.target.shape[0],  self.time_length), device=torch.device('cuda'))
        self.left_vol = torch.zeros(( self.target.shape[0],  self.time_length+1), device=torch.device('cuda'))
        self.left_vol = self.left_vol.detach()
        self.left_vol[:,0] = 10000

    def generate_trading_plan(self, t, prediction):
        """
        Generate trading plans based on predictions at each time step `t`.

        Args:
            t (int): Current time index.
            prediction (torch.Tensor): The predicted values at time step `t`.
        
        Returns:
            tuple: A tuple containing loss_pred, trading_cost, prediction_store, and trading_plan.
        """
        torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

        self.prediction =  prediction.clone()
        #self.prediction = torch.exp( prediction.clone() * self.std + self.mean) 
        self.update_prediction_store(t)
        self.update_trading_plan(t)
        self.update_left_vol(t)
        self.update_cost_x_generation_mpc(t)
        
    def update_prediction_store(self, t):
        """
        Update the prediction store by inserting new predictions for the current time step `t`.

        Args:
            t (int): Current time index.
        """
        self.prediction_store[:, t, t:] = self.prediction[:, :self.time_length- t].clone()

    
    def update_trading_plan(self, t):
        """
        Update the trading plan for the current time step `t` by calculating the optimal decision volume across future steps based on the predictions.

        Args:
            t (int): Current time index.
        """
        # last case, trading left volume.
        if t==self.time_length-1 :
            self.trading_plan[:, t, t:] = self.left_vol[:, t].unsqueeze(1).clone()
        # Otherwise, follow the optimal decision based on the predicitons
        else :
            prediction_trcost = (self.prediction_store[:, t, t:]).clone()  # Clone the tensor to avoid in-place modification
            
            prediction_trcost_sum = torch.sum(prediction_trcost, axis=1, keepdim=True).clone()
            left = self.left_vol[:, t].unsqueeze(1).clone()
            trading_plan =  (prediction_trcost / prediction_trcost_sum) * left
            self.trading_plan[:, t, t:] = trading_plan
            
            
    def update_left_vol(self, t):
        """
        Update the remaining volume for the next time step based on the current trading plan.

        Args:
            t (int): Current time index.
        """
        self.left_vol[:, t + 1] = self.left_vol[:, t] - self.trading_plan[:, t, t]
    
    def update_cost_x_generation_mpc(self, t):
        """
        Update the cost for Model Predictive Control (MPC) based on the current trading plan.

        Args:
            t (int): Current time index.
        """
        trading_plan_square = torch.pow(self.trading_plan[:, t, t].clone(), 2)
        target_plus_epsilon = (self.target_no_sw[:, t].clone())
        division_result = trading_plan_square / target_plus_epsilon
        self.cost_x_generation_mpc[:, t] = division_result

    def return_left_vol_input(self,t):
        return self.left_vol[:,t].clone()

    def return_cost_x(self):
        return self.cost_x_generation_mpc

    def return_prediction(self):
        return self.prediction_store