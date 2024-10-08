import torch
from lightning_module import DFLModelModule
from data_provider.DFLDataModule import initialize_data_module
from utils import *
from trainer import Trainer
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from model import *
import torch

class DFL_Trainer:
    def __init__(self, save_dir, model, args, ticker, weight):
        self.save_dir =save_dir
        self.model = model
        self.args = args
        self.ticker = ticker
        self.weight= weight
        self.accelerator = args.accelerator
        self.learning_rate = args.learning_rate
        
        #for not closed solution
    
        """
        x = cp.Parameter(self.args.prediction_length)
        y = cp.Variable(self.args.prediction_length)
        left_vol = cp.Parameter()
        H = torch.eye(self.args.prediction_length)
        t = cp.Parameter(self.args.prediction_length)

        obj = cp.quad_form( cp.multiply( x,y), H ) 
        cons = [cp.sum(y) == left_vol, y >= 0, cp.multiply(y, t) == 0]
        prob = cp.Problem(cp.Minimize(obj), cons)
        
        
        self.e2e_layer = CvxpyLayer(prob, parameters=[x, left_vol, t], variables=[y])  
        """
        
        self.adjustment_layer = torch.nn.ModuleList()
        with torch.no_grad():
            self.adjustment_layer.append(torch.nn.Linear(args.prediction_length, args.prediction_length))
            self.initialize_weights(self.adjustment_layer[0])
            self.adjustment_layer.append( torch.nn.ReLU() )
            self.adjustment_layer.append(torch.nn.Linear(args.prediction_length, args.prediction_length))
            self.initialize_weights(self.adjustment_layer[2])
                
    def initialize_weights(self, model):
        weight = torch.eye(self.args.prediction_length)  # Identity matrix
        model.weight.copy_(weight)
        model.bias.fill_(0)
        
    def initialize_data_module(self) :
        train_lodaer, val_lodaer, test_loader, scaler = initialize_data_module(self.args, self.ticker)

        return train_lodaer, val_lodaer, test_loader, scaler
    
    def train_model(self):
        
        # Initialize trading module
        self.preds = None
        self.trgs = None
        
        self.train_loader, self.val_loader, self.test_loader, self.scaler = self.initialize_data_module()
        self.module = DFLModelModule.LightningDFLModel(self.model, self.adjustment_layer, self.scaler, self.args.prediction_length, self.learning_rate, self.weight , self.train_loader, self.val_loader, self.test_loader)

        # Use the train_model function from the trainer module
        self.trainer = Trainer(self.module, self.args, self.ticker)

        # Call train_model method
        val_loss = self.trainer.train_model(
            accelerator=self.accelerator,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            max_epochs=self.args.max_epochs,
            dirpath=None  # Change this to the desired directory path if needed
        )

        predictor = self.trainer.test_model(self.test_loader)
        # Retrieve predictions after training
        self.prediction, self.target = self.module.return_prediction()

        if self.preds ==None :
            self.preds = self.prediction.contiguous().view(1,-1)
            self.trgs = self.target.contiguous().view(1,-1)
        else :
            self.preds = torch.cat([self.preds, self.prediction.contiguous().view(1,-1)], dim=0) 
            self.trgs = torch.cat([self.trgs, self.target.contiguous().view(1,-1)], dim=0) 
            
        return self.preds, self.trgs
        
    def save_prediction(self) :
        print(self.preds.shape)
        save_to_csv( self.preds.reshape(1,-1).detach().clone() , "DFL_" + self.model.__class__.__name__, self.save_dir)   
        
    def save_model_parameters(self) :
        # Save the model's state_dict to the specified file path
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "DFL_" + f'{self.model.__class__.__name__}_params.pth'))

    #def load_model_parameters() :

def training_dfl_models(save_dir,  model_list, args, configurations, ticker, weight) :
    pred_results = []
    models = []
    # no -pre training 
    if not model_list :
        for model_name, config in configurations.items():
                model_class = eval(model_name)
                models.append( model_class(config) )
    else : 
        models = model_list
        
    for model in models :
        trainer = DFL_Trainer(save_dir, model, args, ticker, weight)
        pred, trg = trainer.train_model()
        trainer.save_model_parameters()
        trainer.save_prediction()

    return models, pred_results