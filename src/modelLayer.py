import torch
import torch.nn as nn
from transformers import XLMRobertaModel

class FFNN(torch.nn.Module):
    def __init__(self, args):
        super(FFNN, self).__init__()

        self.hidden = args.hidden_dim
        self.device = args.device
        self.embed_dim = args.embed_dim
        self.ffnn =nn.Sequential(nn.Linear(self.embed_dim*2, self.hidden),
                                    nn.Tanh(),
                                    nn.Linear(self.hidden, 1,
                                    bias=True))
        self.ffnn = self.ffnn.to(self.device)

    def forward(self, input):

        output = self.ffnn(input)        
        return output
        



class Bert_Layer(torch.nn.Module):
    def __init__(self, args):
        super(Bert_Layer, self).__init__()
        self.bert_layer = XLMRobertaModel.from_pretrained(args.model_name_or_path)
        self.device = args.device
        self.bert_layer = self.bert_layer.to(self.device)

    def forward(self, **kwargs):
        bert_output = self.bert_layer(input_ids=kwargs['text'].to(self.device), attention_mask=kwargs['mask'].to(self.device), output_hidden_states=True)

        return bert_output


class Mean_Pooling(torch.nn.Module):
    def __init__(self, args):
        super(Mean_Pooling, self).__init__()
        self.batchsize = args.batchsize
        self.max_tok_len = args.max_tok_len
        self.device = args.device

    def forward(self, last_hidden_state, **kwargs):
        # attention_mask  = torch.full((last_hidden_state.shape[0], self.max_tok_len), 1).to(self.device)
        input_mask_expanded = kwargs['mask'].to(self.device).unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings