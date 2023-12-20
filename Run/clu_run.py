import os
import random
import time
import torch
import yaml
import numpy as np

from AllUtils import clusterUtils, test_method, dataloaderUtils

from Coordinator.server_model import FMDTC_server
from Participants.client_model import FMDTC_client

from AllUtils.LossFunction import ClaLossFun, PreLossFun
from Participants.participant import Client


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)


setup_seed(1933)
config = yaml.safe_load(open('config.yaml'))

client_num = config["client_server"]["client_num"]
embedding_size = config["embedding_size"]
device = "cuda:" + str(config["cuda"])

nhead = config["transformer"]["nhead"]
num_layers = config["transformer"]["num_layers"]
dropout = config["transformer"]["dropout"]
dim_feedforward = config["transformer"]["dim_feedforward"]
dataset = str(config["dataset"])
distance_type = str(config["distance_type"])
test_batch = config["test_batch"]

train_batch = config["cluster"]["train_batch"]
epochs = config["cluster"]["epochs"]
early_stop = config["cluster"]["early_stop"]
learning_rate = config["cluster"]["learning_rate"]
classifier_rate = config["cluster"]["classifier_rate"]
weight_decay = config["cluster"]["weight_decay"]
classifier_weight_decay = config["cluster"]["classifier_weight_decay"]
k = config["cluster"]["k"]
a = config["cluster"]["a"]
classifier_train = config["cluster"]["classifier_train"]
s = config["space"]
t=config["time"]
d=config["speed"]
cluster_size = embedding_size * (s+t+d)
# model setting

# =====================coordinator===========================
class Server(object):
    def __init__(self):
        pass

    def updata_net_w(self):
        pass

    def train_server(self, H_a, H_p, H_n, H_c, L_a, L_p, L_n, L_c, batch_index_p, batch_index_c, optimizer, optimizer_c,
                     maxlen, maxlen_c, net_glob_server, closs, ploss, epoch):

        if epoch < classifier_train:
            y_predict = net_glob_server(H_c, L_c, maxlen_c, cluster=True)
            H_c.retain_grad()
            l = closs(y_predict, batch_index_c)
            optimizer_c.zero_grad()
            l.backward()
            optimizer_c.step()
            dH_c = H_c.grad.clone().detach()
            return [dH_c], l
        else:
            V_a = net_glob_server(H_a, L_a, maxlen[0])
            V_p = net_glob_server(H_p, L_p, maxlen[1])
            V_n = net_glob_server(H_n, L_n, maxlen[2])
            y_predict = net_glob_server(H_c, L_c, maxlen_c, cluster=True)
            H_a.retain_grad()
            H_p.retain_grad()
            H_n.retain_grad()
            H_c.retain_grad()
            l = a*closs(y_predict, batch_index_c) + (1-a)*ploss(V_a, V_p, V_n, batch_index_p)
            optimizer.zero_grad()
            optimizer_c.zero_grad()
            l.backward()
            optimizer.step()
            optimizer_c.step()
            dH_a = H_a.grad.clone().detach()
            dH_p = H_p.grad.clone().detach()
            dH_n = H_n.grad.clone().detach()
            dH_c = H_c.grad.clone().detach()
            return [dH_a, dH_p, dH_n, dH_c], l


# =====================coordinator===========================


def train(load_client_model=None, load_server_model=None):
    if load_client_model == None or load_server_model == None:
        print("No model")
        return
    
    dataload = dataloaderUtils.DataLoader()
    dataload.get_triplets()
    dataload.triplet_groud_truth()
    numofbatch = int(dataload.return_triplets_num() / train_batch)
    batchload = dataloaderUtils.BatchLoader(train_batch)
    roadnet = dataloaderUtils.load_netowrk(dataset)

    net_glob_client = FMDTC_client(embedding_size, device,s=s,t=t,d=d)
    net_glob_server = FMDTC_server(cluster_size, embedding_size, k, device, nhead, num_layers, dropout, dim_feedforward, f_num=s+d+t)
    net_glob_client = net_glob_client.double().to(device)
    net_glob_server = net_glob_server.double().to(device)
    server_model = torch.load(load_server_model, map_location=device)
    for i in list(server_model.keys()):
        if i.startswith('classifier'):
            del server_model[i]
    net_glob_client.load_state_dict(torch.load(load_client_model, map_location=device))
    net_glob_server.load_state_dict(server_model, strict=False)
    prels = PreLossFun(train_batch).to(device)
    cluls = ClaLossFun(train_batch).to(device)
    net_glob_client_w = net_glob_client.state_dict()
    client_list = [Client(id, net_glob_client_w, embedding_size, device, classifier_train, s=s, t=t, d=d) for id in range(client_num)]
    server = Server()

    optimizer_client = torch.optim.Adam([p for p in net_glob_client.parameters() if p.requires_grad],
                                        lr=learning_rate,
                                        weight_decay=weight_decay)
    optimizer_server = torch.optim.Adam([p for p in net_glob_server.drop.parameters() if p.requires_grad] +
                                        [p for p in net_glob_server.MF2v.parameters() if p.requires_grad],
                                        lr=learning_rate,
                                        weight_decay=weight_decay)
    optimizer_classifier = torch.optim.Adam([p for p in net_glob_server.classifier.parameters() if p.requires_grad],
                                            lr=classifier_rate,
                                            weight_decay=classifier_weight_decay)

    best_NMI = 0
    lastepoch = '0'

    print("=======================================start cluster train=======================================")
    for epoch in range(int(lastepoch), epochs):

        net_glob_client.eval()
        net_glob_server.eval()
        with torch.no_grad():
            # Get pseudo labels
            node_train, _, t2vec_train, speed_train = dataload.load("train")
            embedding_train = test_method.compute_embedding(roadnet=roadnet, net_c=net_glob_client,
                                                            net_s=net_glob_server,
                                                            test_traj=list(node_train),
                                                            test_time=list(t2vec_train),
                                                            test_speed=list(speed_train),
                                                            test_batch=test_batch)
            clusterUtils.pseudoLabels(k, embedding_train.cpu())

        # save model
        if epoch % 1 == 0:
            net_glob_client.eval()
            net_glob_server.eval()
            with torch.no_grad():
                vali_node_list, _, vali_d2vec_list, vali_speed_list = dataload.load(load_part='test')
                s6 = time.time()
                embeddings = test_method.compute_embedding(roadnet=roadnet, net_c=net_glob_client,
                                                               net_s=net_glob_server,
                                                               test_traj=list(vali_node_list),
                                                               test_time=list(vali_d2vec_list),
                                                               test_speed=list(vali_speed_list),
                                                               test_batch=test_batch)
                acc = test_method.test_model(embeddings, isvali=False)
                s_score, NMI, AMI, ARS, FMS = test_method.test_cluster(embeddings, k, "test")

                print("test time: ", time.time() - s6,'  epoch:', epoch)
                print(acc[0], acc[1], acc[2])
                print(NMI, AMI, ARS, FMS)
                print("")

                # save model
                save_model_client = 'Models/Clu_Models/{}/{}/participants/participant_checkpoint.pkl'.format(dataset, distance_type)
                torch.save(net_glob_client.state_dict(), save_model_client)
                save_model_server = 'Models/Clu_Models/{}/{}/coordinator/coordinator_checkpotin.pkl'.format(dataset, distance_type)
                torch.save(net_glob_server.state_dict(), save_model_server)
                if NMI > best_NMI:
                    best_NMI = NMI
                    best_NMI_epoch = epoch
                    best_NMI_client = 'Models/Clu_Models/{}/{}/participants/participant_NMI_BEST.pkl'.format(dataset, distance_type)
                    torch.save(net_glob_client.state_dict(), best_NMI_client)
                    best_NMI_server = 'Models/Clu_Models/{}/{}/coordinator/coordinator_NMI_BEST.pkl'.format(dataset, distance_type)
                    torch.save(net_glob_server.state_dict(), best_NMI_server)
                if epoch - best_NMI_epoch >= early_stop:
                    break
        
        # train
        node_train, _, t2vec_train, speed_train = dataload.load("train")
        net_glob_client.train()
        net_glob_server.train()
        s1 = time.time()
        start = len(node_train)
        for bt in range(numofbatch):
            if start - train_batch < 0:
                start = len(node_train)
            batch_index_clu = list(range(start - train_batch, start))
            start -= train_batch
            node_list = node_train[batch_index_clu]
            time_list = t2vec_train[batch_index_clu]
            speed_list = speed_train[batch_index_clu]
            maxlen_c = max(list(map(len, node_list)))
            C = [node_list, time_list, speed_list]
            X = batchload.getbatch_one()
            maxlen = [max(list(map(len, X[0]))), max(list(map(len, X[3]))), max(list(map(len, X[6])))]
            client_data_num = train_batch / client_num  
            h_a, h_p, h_n, h_c, l_a, l_p, l_n, l_c = [], [], [], [], [], [], [], []

            client_data_index = []
            for id in range(client_num):
                last = 0
                if id == client_num - 1:
                    last = 1
                Xid, a1, a2 = batchload.getclientdata(X, id, client_data_num, last, train_batch)
                Cid, _, _ = batchload.getclientdata(C, id, client_data_num, last, train_batch)
                # The data owned by each client

                client_data_index.append([a1, a2])
                a, p, n = client_list[id].forward(roadnet=roadnet, data=Xid, maxlen=maxlen)
                c = client_list[id].forward_c(roadnet=roadnet, data=Cid, maxlen=maxlen_c)
                # Complete client forward propagation

                h_a.append(a[0])
                h_p.append(p[0])
                h_n.append(n[0])
                h_c.append(c[0])
                l_a.append(a[1])
                l_p.append(p[1])
                l_n.append(n[1])
                l_c.append(c[1])
            H_a = torch.cat(h_a, 0)
            H_p = torch.cat(h_p, 0)
            H_n = torch.cat(h_n, 0)
            H_c = torch.cat(h_c, 0)
            L_a = torch.cat(l_a, 0)
            L_p = torch.cat(l_p, 0)
            L_n = torch.cat(l_n, 0)
            L_c = torch.cat(l_c, 0)
            # Aggregating data
            
            dH, loss = server.train_server(H_a, H_p, H_n, H_c, L_a, L_p, L_n, L_c, X[9],
                                                                   batch_index_clu,
                                                                   optimizer=optimizer_server,
                                                                   optimizer_c=optimizer_classifier,
                                                                   maxlen=maxlen,
                                                                   maxlen_c=maxlen_c,
                                                                   net_glob_server=net_glob_server,
                                                                   closs=cluls,
                                                                   ploss=prels,
                                                                   epoch=epoch)


            for id in range(client_num):
                if id == 0:
                    a1 = client_data_index[id][0]
                    a2 = client_data_index[id][1]
                    td = client_list[id].calculate_gradient_clu(dH, a1, a2, epoch)
                    for i, param in enumerate(net_glob_client.parameters()):
                        param.grad = td[i]
                else:
                    a1 = client_data_index[id][0]
                    a2 = client_data_index[id][1]
                    td = client_list[id].calculate_gradient_clu(dH, a1, a2, epoch)
                    for i, param in enumerate(net_glob_client.parameters()):
                        param.grad = param.grad + td[i]
            # Simulate the cumulative model gradient at the aggregator
                        
            if epoch >= classifier_train:
                optimizer_client.step()
            optimizer_client.zero_grad()
            # Complete the remaining backpropagation

            for id in range(client_num):
                client_list[id].updata_net_w(net_glob_client.state_dict())
            # Synchronize participant models

        print("train time: ", time.time() - s1, "    loss:", loss.item())

    print("the best epoch:", best_NMI_epoch, "the best NMI:", best_NMI)

    print("=======================================end pre train=======================================")


def test(load_client_model=None, load_server_model=None):
    if load_client_model != None and load_server_model != None:
        dataload = dataloaderUtils.DataLoader()
        net_glob_client = FMDTC_client(embedding_size, device, s=s,t=t,d=d)
        net_glob_server = FMDTC_server(cluster_size, embedding_size, k, device, nhead, num_layers, dropout,
                                       dim_feedforward, f_num=d+t+s)
        net_glob_client = net_glob_client.double().to(device)
        net_glob_server = net_glob_server.double().to(device)
        server_model = torch.load(load_server_model, map_location=device)
        for i in list(server_model.keys()):
            if i.startswith('classifier'):
                del server_model[i]
        net_glob_client.load_state_dict(torch.load(load_client_model, map_location=device))
        net_glob_server.load_state_dict(server_model, strict=False)
        roadnet = dataloaderUtils.load_netowrk(dataset)
        net_glob_client.eval()
        net_glob_server.eval()
        with torch.no_grad():
            s6 = time.time()
            vali_node_list, _, vali_d2vec_list, vali_speed_list = dataload.load(load_part='test')
            t1 = time.time()
            embeddings = test_method.compute_embedding(roadnet=roadnet, net_c=net_glob_client,
                                                           net_s=net_glob_server,
                                                           test_traj=list(vali_node_list),
                                                           test_time=list(vali_d2vec_list),
                                                           test_speed=list(vali_speed_list),
                                                           test_batch=test_batch)
            print("embedding time:", time.time()-t1)
            s7 = time.time()
            print("test time: ", s7 - s6)
            NMI = []
            ss = []
            for i in range(0, 20):
                s_score, NMI_score, AMI, ARS, FMS = test_method.test_cluster(embeddings, k, "test")
                NMI.append(NMI_score)
                ss.append(s_score)
                print(s_score, NMI_score)
            print("NMI_std", np.std(NMI))
            print("NMI_mean", np.mean(NMI))
            print("ss_std", np.std(ss))
            print("ss_mean", np.mean(ss))
