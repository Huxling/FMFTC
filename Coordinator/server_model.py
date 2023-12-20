import math
import torch
import torch.nn as nn
from torch.nn.modules import transformer
import torch.nn.functional as F

class PositionalEncoding():
    def __init__(self, d_model, device):
        self.d_model = d_model
        self.device = device
    def getpe(self, max_len=10):

        pe = torch.zeros(max_len, self.d_model, dtype=torch.double).to(self.device)
        position = torch.arange(0, max_len).unsqueeze(1)  # max_len * 1

        # 1/10000^(2k/d) = e^(2k/d * ln 10000)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) *  # 1 * d_model
                             -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)      # max_len * d_model
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # 1 * max_len * d_model,

        return pe

class MFEmbedding(nn.Module):
    def __init__(self, d_model, model_num, nhead, num_layers, device, dropout, dim_feedforward):
        super(MFEmbedding, self).__init__()
        self.device = device
        self.transformerEncoderLayer = transformer.TransformerEncoderLayer(d_model=d_model*model_num,
                                                                           nhead=nhead,
                                                                           batch_first=True,
                                                                           device=self.device,
                                                                           dropout=dropout,
                                                                           dim_feedforward=dim_feedforward,
                                                                           activation="relu")
        self.transformerEncoder = transformer.TransformerEncoder(self.transformerEncoderLayer, num_layers=num_layers)
        self.positionalEncoding = PositionalEncoding(d_model=d_model*model_num, device=self.device)

    def getMask(self, s_input, seq_lengths, maxlen):
        mask = torch.zeros((s_input.shape[0], maxlen), dtype=torch.double).to(self.device)
        for i, l in enumerate(seq_lengths):
            if l < maxlen:
                mask[i, l:] = 1
        return mask

    def forward(self, x, seq_lengths, maxlen):
        mask = self.getMask(x, seq_lengths, maxlen)
        pe = self.positionalEncoding.getpe(max_len=maxlen)
        x = x + pe
        x = self.transformerEncoder(src=x, src_key_padding_mask=mask)
        x = x[:, 0, :]
        return x

class Network_server(nn.Module):
    def __init__(self, embedding_size, device, nhead, num_layers, dropout, dim_feedforward, f_num):
        super(Network_server, self).__init__()
        self.MF_Embedding = MFEmbedding(embedding_size, f_num, nhead, num_layers, device, dropout, dim_feedforward)
    def forward(self,x, seq_lengths, maxlen):
        output = self.MF_Embedding(x, seq_lengths, maxlen)
        return output

class FMDTC_server(nn.Module):
    def __init__(self, cluster_size, embedding_size, k, device, nhead, num_layers, dropout, dim_feedforward, f_num):
        super(FMDTC_server, self).__init__()
        self.drop = nn.Dropout(0.1)
        self.MF2v = Network_server(embedding_size=embedding_size,
                           device=device,
                           nhead=nhead,
                           num_layers=num_layers,
                           dropout=dropout,
                           dim_feedforward=dim_feedforward,
                                   f_num=f_num)
        self.classifier = nn.Sequential(
                            nn.Linear(cluster_size, k)).to(device)
        
    def forward(self, x, seq_lengths, maxlen, cluster=False):
        x = self.drop(x)
        x = self.MF2v(x, seq_lengths, maxlen)
        if cluster == False:
            return x
        y = self.classifier(x)
        return y