import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, drop_out=0.3):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop_out = drop_out
        
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=self.drop_out)

    def forward(self, x):
        output, hidden = self.rnn(x)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, drop_out=0.3):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop_out = drop_out
        
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=self.drop_out)
        self.linear = nn.Linear(hidden_size, self.input_size)

    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)
        output = self.linear(output)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, configs):
        super(Seq2Seq, self).__init__()
        self.features = configs.features
        self.pred_len = configs.pred_len
        self.hidden_dim = configs.hidden_dim
        self.drop_out = configs.drop_out
        self.num_layers = configs.num_layers

        self.encoder = Encoder(self.features, self.hidden_dim , self.num_layers, self.drop_out)
        self.decoder = Decoder(self.features, self.hidden_dim , self.num_layers, self.drop_out)
        self.device = configs.device

    def forward(self, input, target, teacher_forcing_ratio=0.3):
        # input_size : batch, context, feature_size
        # pred_len : batch, prediction_len, feature_size
        batch_size = input.shape[0]
        target_feature = target.shape[2]
        
        if next(self.encoder.parameters()).device.type == 'cuda':
            outputs = torch.zeros(batch_size,  self.pred_len,  target_feature).to(self.device)
        else:
            outputs = torch.zeros(batch_size,  self.pred_len,  target_feature)
        _, hidden = self.encoder(input)

        decoder_input = input[:, -1, :]

        for t in range(self.pred_len):
            decoder_input = decoder_input.unsqueeze(1)
            output, hidden = self.decoder(decoder_input, hidden)
            output =  output.squeeze(1)
            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input = target[:, t] if teacher_force else output
            outputs[:,t,:] = output

        return outputs

    def predict(self, input, target):
        batch_size = input.shape[0]
        target_feature = target.shape[2]

        outputs = torch.zeros(batch_size, self.pred_len, target_feature).to(self.device)

        _, hidden = self.encoder(input)

        decoder_input = input[:, -1, :]

        for t in range(self.pred_len):
            decoder_input = decoder_input.unsqueeze(1)
            output, hidden = self.decoder(decoder_input, hidden)
            output =  output.squeeze(1)
            decoder_input = output
            outputs[:,t,:] = output
            
        return outputs