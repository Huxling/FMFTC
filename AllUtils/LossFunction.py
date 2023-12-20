import pickle
from torch.nn import Parameter
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch
import numpy as np
import math
import yaml
import pandas as pd

class PreLossFun(nn.Module):
    def __init__(self, train_batch):
        super(PreLossFun, self).__init__()
        self.train_batch = train_batch
        config = yaml.safe_load(open('config.yaml'))
        self.triplets_dis = np.load(str(config["path_triplets_truth"]))

    def forward(self, embedding_a, embedding_p, embedding_n, batch_index):
        batch_triplet_dis = self.triplets_dis[batch_index]
        batch_loss = 0.0
        for i in range(self.train_batch):
            D_ap = math.exp(-batch_triplet_dis[i][0])
            D_an = math.exp(-batch_triplet_dis[i][1])
            v_ap = torch.exp(-torch.dist(embedding_a[i], embedding_p[i], p=2))
            v_an = torch.exp(-torch.dist(embedding_a[i], embedding_n[i], p=2))
            loss_entire_ap = D_ap * ((D_ap - v_ap) ** 2)
            loss_entire_an = D_an * ((D_an - v_an) ** 2)
            oneloss = loss_entire_ap + loss_entire_an
            batch_loss += oneloss
        mean_batch_loss = batch_loss / self.train_batch
        return mean_batch_loss

class ClaLossFun(nn.Module):
    def __init__(self, train_batch):
        super(ClaLossFun, self).__init__()
        self.train_batch = train_batch
        self.loss = CrossEntropyLoss()
        self.config = yaml.safe_load(open('config.yaml'))
        self.device = "cuda:" + str(self.config["cuda"])

    def forward(self, y_predict, batch_index_c):
        y = np.array(pickle.load(open(str(self.config["pseudo_labels"]), 'rb')))
        y = torch.tensor(y[batch_index_c]).to(self.device)
        clu_loss = self.loss(y_predict, y.long())
        return clu_loss