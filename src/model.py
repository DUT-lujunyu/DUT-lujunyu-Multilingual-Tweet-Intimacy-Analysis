import torch
from src.modelLayer import *
import torch.nn as nn

class Baseline(torch.nn.Module):
    def __init__(self, args):
        super(Baseline, self).__init__()

        self.dropout = nn.Dropout(args.dropout)
        self.device = args.device
        self.bert_layer = Bert_Layer(args)
        self.ffnn = FFNN(args)
        self.pooling = Mean_Pooling(args)
        self.bert_layer = self.bert_layer.to(self.device)
        self.ffnn = self.ffnn.to(self.device)
        self.pooling = self.pooling.to(self.device)

    def forward(self, **kwargs):
        
        bert_output = self.bert_layer(**kwargs)
        temp = bert_output[0]
        temp = self.pooling(temp, **kwargs)
        temp = self.dropout(temp)
        output = self.ffnn(torch.concat((bert_output.pooler_output, temp), dim=-1))
        return output
