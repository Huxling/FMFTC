import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

class GAT(nn.Module):
    def __init__(self, embedding_size, device):
        super(GAT, self).__init__()
        self.device = device
        self.conv1 = GATConv(embedding_size, 8, heads=8)
        # self.drop = nn.Dropout(0.5)

    def forward(self, data):
        x = data[0].x.to(self.device)
        adj = data[1].to(self.device)
        x = F.relu(self.conv1(x, adj))
        # x = self.drop(x)
        # (num_nodes, embedding_size)
        return x

class TrajEmbedding(nn.Module):
    def __init__(self, embedding_size, device):
        super(TrajEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.device = device
        self.gat = GAT(embedding_size, device).to(self.device)

    def forward(self, network, traj_seqs, maxlen):
        batch_size = len(traj_seqs)
        seq_lengths = list(map(len, traj_seqs))
        # print(seq_lengths)

        for traj_one in traj_seqs:
            traj_one += [0]*(maxlen-len(traj_one))
        for i, traj_one in enumerate(traj_seqs):
            traj_seqs[i] = traj_one[:maxlen]

        embedded_seq_tensor = torch.zeros((batch_size, maxlen, self.embedding_size), dtype=torch.double)
        # （batch * maximum length * embedding）

        seq_lengths = torch.LongTensor(seq_lengths).to(self.device)
        traj_seqs = torch.tensor(traj_seqs).to(self.device)

        node_embeddings = self.gat(network)
        for idx, (seq, seqlen) in enumerate(zip(traj_seqs, seq_lengths)):
            embedded_seq_tensor[idx, :seqlen] = node_embeddings.index_select(0, seq[:seqlen])

        # move to cuda device
        seq_lengths = seq_lengths.cpu()
        embedded_seq_tensor = embedded_seq_tensor.to(self.device)
        return embedded_seq_tensor, seq_lengths


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_size, device):
        super(TimeEmbedding, self).__init__()
        self.device = device
        self.embedding_size = embedding_size

    def forward(self, time_seqs, maxlen):
        batch_size = len(time_seqs)
        seq_lengths = list(map(len, time_seqs))
        for time_one in time_seqs:
            time_one += [[0 for i in range(self.embedding_size)]]*(maxlen-len(time_one))

        for i, time_one in enumerate(time_seqs):
            time_seqs[i] = time_one[:maxlen]

        # prepare sequence tensor
        embedded_seq_tensor = torch.zeros((batch_size, maxlen, self.embedding_size), dtype=torch.double)

        seq_lengths = torch.LongTensor(seq_lengths).to(self.device)

        vec_time_seqs = torch.tensor(time_seqs).to(self.device)

        # get embedding for trajectory embeddings
        for idx, (seq, seqlen) in enumerate(zip(vec_time_seqs, seq_lengths)):
            embedded_seq_tensor[idx, :seqlen] = seq[:seqlen]

        # move to cuda device
        embedded_seq_tensor = embedded_seq_tensor.to(self.device)
        seq_lengths = seq_lengths.cpu()
        return embedded_seq_tensor, seq_lengths


class SpeedEmbedding(nn.Module):
    def __init__(self, embedding_size, device):
        super(SpeedEmbedding,self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.fline = nn.Linear(3, embedding_size).to(self.device)
        self.FFN = nn.Sequential(
            nn.Linear(embedding_size, int(embedding_size*0.5)),
            nn.ReLU(),
            nn.Linear(int(embedding_size*0.5), embedding_size),
            # nn.Dropout(0.1)
        ).to(self.device)
    def forward(self, speed_seqs, maxlen):

        seq_lengths = list(map(len, speed_seqs))
        for speed_one in speed_seqs:
            speed_one += [0] * (maxlen - len(speed_one))

        for i, speed_one in enumerate(speed_seqs):
            speed_seqs[i] = speed_one[:maxlen]
        # padding

        vec_speed_seqs = torch.tensor(speed_seqs, dtype=torch.double)
        seq_lengths = torch.LongTensor(seq_lengths).to(self.device)
        # print(vec_speed_seqs)

        avg = torch.sum(vec_speed_seqs, dim=1)
        # maxs = torch.max(vec_speed_seqs, dim=1)[0]
        # mins = torch.max(vec_speed_seqs, dim=1)[0]
        for i in range(0, avg.shape[0]):
            avg[i] = avg[i] / seq_lengths[i]
        # average (batch, 1)

        dis1 = F.softmax(vec_speed_seqs, dim=-1)
        # dis2 = F.softmax(-vec_speed_seqs, dim=-1)
        t = speed_seqs
        for i, x in enumerate(t):
            for j, _ in enumerate(x):
                t[i][j] = [dis1[i][j], avg[i], vec_speed_seqs[i][j]]
        z = torch.DoubleTensor(speed_seqs).to(self.device)

        z = self.FFN(self.fline(z))
        seq_lengths = seq_lengths.cpu()
        return z, seq_lengths

class Co_Att(nn.Module):
    def __init__(self, dim, t, s, d):
        super(Co_Att, self).__init__()
        self.s = s
        self.t = t
        self.d = d
        self.f_num = s+t+d
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.temperature = dim ** 0.5
        self.FFN = nn.Sequential(
            nn.Linear(dim, int(dim*0.5)),
            nn.ReLU(),
            nn.Linear(int(dim*0.5), dim),
            # nn.Dropout(0.1)
        )
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, seq_s, seq_t, seq_d):

        if self.f_num == 2 and self.s == 0:
            h = torch.stack([seq_t, seq_d], 2)  # [b, n, 2, dim]
        elif self.f_num == 2 and self.t == 0:
            h = torch.stack([seq_s, seq_d], 2)  # [b, n, 2, dim]
        elif self.f_num == 2 and self.d == 0:
            h = torch.stack([seq_s, seq_t], 2)  # [b, n, 2, dim]
        else:
            h = torch.stack([seq_s, seq_t, seq_d], 2)  # [b, n, 3, dim]
        q = self.Wq(h)
        k = self.Wk(h)
        v = self.Wv(h)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = F.softmax(attn, dim=-1)
        attn_h = torch.matmul(attn, v)

        attn_o = self.FFN(attn_h) + attn_h
        attn_o = self.layer_norm(attn_o)
        if self.f_num == 2 and self.s == 0:
            att_s = 0
            att_t = attn_o[:, :, 0, :]
            att_d = attn_o[:, :, 1, :]
        elif self.f_num == 2 and self.t == 0:
            att_s = attn_o[:, :, 0, :]
            att_t = 0
            att_d = attn_o[:, :, 1, :]
        elif self.f_num == 2 and self.d == 0:
            att_s = attn_o[:, :, 0, :]
            att_t = attn_o[:, :, 1, :]
            att_d = 0
        else:
            att_s = attn_o[:, :, 0, :]
            att_t = attn_o[:, :, 1, :]
            att_d = attn_o[:, :, 2, :]
        return att_s, att_t, att_d


class Network_client(nn.Module):
    def __init__(self, embedding_size, device, s, t, d):
        super(Network_client, self).__init__()
        self.s = s
        self.t = t
        self.d = d
        if s == 1:
            self.S_Embedding = TrajEmbedding(embedding_size, device)
        if t == 1:
            self.T_Embedding = TimeEmbedding(embedding_size, device)
        if d == 1:
            self.D_Embedding = SpeedEmbedding(embedding_size, device)
        if d+t+s != 1:
            self.co_attention = Co_Att(embedding_size, s=s, t=t, d=d).to(device)
    def forward(self,network, traj_seqs, time_seqs, speed_seqs, maxlen):
        s_input, d_input, t_input = 0, 0, 0
        if self.s == 1:
            s_input, seq_lengths = self.S_Embedding(network, traj_seqs, maxlen)
            if self.t + self.d == 0:
                return s_input, seq_lengths
        if self.d == 1:
            d_input, seq_lengths = self.D_Embedding(speed_seqs, maxlen)
            if self.t + self.s == 0:
                return d_input, seq_lengths
        if self.t == 1:
            t_input, seq_lengths = self.T_Embedding(time_seqs, maxlen)
            if self.s + self.d == 0:
                return t_input, seq_lengths
        s_co, t_co, d_co = self.co_attention(s_input, t_input, d_input)
        if self.s == 0:
            smashed_data = torch.cat((t_co, d_co), dim=2)
        elif self.t == 0:
            smashed_data = torch.cat((s_co, d_co), dim=2)
        elif self.d == 0:
            smashed_data = torch.cat((s_co, t_co), dim=2)
        else:
            smashed_data = torch.cat((s_co, t_co, d_co), dim=2)
        return smashed_data, seq_lengths

class FMDTC_client(nn.Module):
    def __init__(self, embedding_size, device, t, s, d):
        super(FMDTC_client, self).__init__()
        self.MF2v_c = Network_client(embedding_size=embedding_size, device=device, t=t, s=s, d=d)
    def forward(self, network, traj_seqs, time_seqs, speed_seqs, maxlen):
        for trajectory in traj_seqs:
            while trajectory and trajectory[-1] == 0:
                trajectory.pop()
        for speeds in speed_seqs:
            while speeds and speeds[-1] == 0:
                speeds.pop()
        for times in time_seqs:
            while times and times[-1] == 0:
                times.pop()
        smashed_data, seq_lengths = self.MF2v_c(network, traj_seqs, time_seqs, speed_seqs, maxlen)
        return smashed_data, seq_lengths