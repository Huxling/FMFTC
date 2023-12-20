from Participants.client_model import FMDTC_client

class Client(object):
    def __init__(self, idx, init_w, embedding_size, device, classifier_train, s,d,t):
        self.idx = idx
        self.net = FMDTC_client(embedding_size, device, s=s,d=d,t=t)
        self.net.load_state_dict(init_w)
        self.net.double().to(device)
        self.net.train()
        self.classifier_train = classifier_train
    def updata_net_w(self, glob_w):
        self.net.load_state_dict(glob_w)

    def forward_c(self, roadnet, data, maxlen):
        self.c, c_lengths = self.net(roadnet, traj_seqs=data[0], time_seqs=data[1], speed_seqs=data[2],
                                     maxlen=maxlen)

        smashed_c = self.c.clone().detach().requires_grad_(True)

        return [smashed_c, c_lengths]
    def forward(self, roadnet, data, maxlen):
        self.a, a_lengths = self.net(roadnet, traj_seqs=data[0], time_seqs=data[1], speed_seqs=data[2], maxlen=maxlen[0])
        self.p, p_lengths = self.net(roadnet, traj_seqs=data[3], time_seqs=data[4], speed_seqs=data[5], maxlen=maxlen[1])
        self.n, n_lengths = self.net(roadnet, traj_seqs=data[6], time_seqs=data[7], speed_seqs=data[8], maxlen=maxlen[2])

        smashed_a = self.a.clone().detach().requires_grad_(True)
        smashed_p = self.p.clone().detach().requires_grad_(True)
        smashed_n = self.n.clone().detach().requires_grad_(True)

        return [smashed_a, a_lengths], [smashed_p, p_lengths], [smashed_n, n_lengths]

    def calculate_gradient_clu(self, dH, a1, a2, epoch):
        for param in self.net.parameters():
            if param.grad is not None:
                param.grad.zero_()
        if epoch < self.classifier_train:
            self.c.backward(dH[0][a1:a2])
        else:
            self.a.backward(dH[0][a1:a2])
            self.p.backward(dH[1][a1:a2])
            self.n.backward(dH[2][a1:a2])
            self.c.backward(dH[3][a1:a2])
        t = []
        for param in self.net.parameters():
            t.append(param.grad)
        return t

    def calculate_gradient(self, dH_a, dH_p, dH_n):
        for param in self.net.parameters():
            if param.grad is not None:
                param.grad.zero_()
        self.a.backward(dH_a)
        self.p.backward(dH_p)
        self.n.backward(dH_n)
        t = []
        for param in self.net.parameters():
            t.append(param.grad)
        return t

