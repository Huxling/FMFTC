import os
import random
import time
import torch
import yaml
import h5py
import numpy as np
from AllUtils import test_method, dataloaderUtils
from Coordinator.server_model import FMDTC_server
from Participants.client_model import FMDTC_client
from AllUtils.LossFunction import PreLossFun
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

epochs = config["epochs"]
client_num = config["client_server"]["client_num"]
embedding_size = config["embedding_size"]
device = "cuda:" + str(config["cuda"])
cluster_size = embedding_size * 3
k = config["cluster"]["k"]
nhead = config["transformer"]["nhead"]
num_layers = config["transformer"]["num_layers"]
dropout = config["transformer"]["dropout"]
dim_feedforward = config["transformer"]["dim_feedforward"]
learning_rate = config["learning_rate"]
weight_decay = config["weight_decay"]
train_batch = config["train_batch"]
dataset = str(config["dataset"])
distance_type = str(config["distance_type"])
test_batch = config["test_batch"]
early_stop = config["early_stop"]
s = config["space"]
t = config["time"]
d = config["speed"]
# model setting

class Server(object):
    def __init__(self):
        pass

    def updata_net_w(self):
        pass

    def train_server(self,H_a, H_p, H_n, L_a, L_p, L_n, batch_index, optimizer, maxlen, net_glob_server, prels):
        V_a = net_glob_server(H_a, L_a, maxlen[0])
        V_p = net_glob_server(H_p, L_p, maxlen[1])
        V_n = net_glob_server(H_n, L_n, maxlen[2])
        H_a.retain_grad()
        H_p.retain_grad()
        H_n.retain_grad()
        loss = prels(V_a, V_p, V_n, batch_index)
        optimizer.zero_grad()
        loss.backward()
        dH_a = H_a.grad.clone().detach()
        dH_p = H_p.grad.clone().detach()
        dH_n = H_n.grad.clone().detach()
        optimizer.step()
        return dH_a, dH_p, dH_n, loss

def train():
    # =====================Load data===========================
    dataload = dataloaderUtils.DataLoader()
    dataload.get_triplets()
    dataload.triplet_groud_truth()
    numofbatch = int(dataload.return_triplets_num() / train_batch)
    batchload = dataloaderUtils.BatchLoader(train_batch)
    roadnet = dataloaderUtils.load_netowrk(dataset)
    # =====================Load data===========================

    # =====================initialization===========================
    net_glob_client = FMDTC_client(embedding_size, device, s=s,t=t,d=d)
    net_glob_server = FMDTC_server(cluster_size, embedding_size, k, device, nhead, num_layers, dropout, dim_feedforward, f_num=t+s+d)
    net_glob_client = net_glob_client.double().to(device)
    net_glob_server = net_glob_server.double().to(device)
    prels = PreLossFun(train_batch).to(device)
    net_glob_client_w = net_glob_client.state_dict()
    client_list = [Client(id, net_glob_client_w, embedding_size, device, 0, s=s,t=t,d=d) for id in range(client_num)]
    server = Server()
    optimizer_client = torch.optim.Adam([p for p in net_glob_client.parameters() if p.requires_grad],
                                        lr=learning_rate,
                                        weight_decay=weight_decay)
    optimizer_server = torch.optim.Adam([p for p in net_glob_server.parameters() if p.requires_grad],
                                        lr=learning_rate,
                                        weight_decay=weight_decay)
    # =====================initialization===========================

    best_HR10 = 0
    best_NMI = 0
    lastepoch = '0'
    # =====================train===========================
    print("pre train")
    for epoch in range(int(lastepoch), epochs):
        net_glob_client.train()
        net_glob_server.train()
        s1 = time.time()

        for bt in range(numofbatch):
            X = batchload.getbatch_one()
            maxlen = [max(list(map(len, X[0]))), max(list(map(len, X[3]))), max(list(map(len, X[6])))]
            client_data_num = train_batch / client_num  # Number of data for each participant in each batch
            h_a, h_p, h_n, l_a, l_p, l_n = [], [], [], [], [], []

            client_data_index = []
            for id in range(client_num):
                last = 0
                if id == client_num - 1:
                    last = 1
                Xid, a1, a2 = batchload.getclientdata(X, id, client_data_num, last, train_batch)
                # The data owned by each client
                client_data_index.append([a1, a2])

                a, p, n = client_list[id].forward(roadnet=roadnet, data=Xid, maxlen=maxlen)
                # Complete client forward propagation

                h_a.append(a[0])
                h_p.append(p[0])
                h_n.append(n[0])
                l_a.append(a[1])
                l_p.append(p[1])
                l_n.append(n[1])
            H_a = torch.cat(h_a, 0)
            H_p = torch.cat(h_p, 0)
            H_n = torch.cat(h_n, 0)
            L_a = torch.cat(l_a, 0)
            L_p = torch.cat(l_p, 0)
            L_n = torch.cat(l_n, 0)
            # Aggregating data

            dH_a, dH_p, dH_n, loss = server.train_server(H_a, H_p, H_n, L_a, L_p, L_n, X[9], optimizer_server, maxlen,
                                                         net_glob_server, prels)
            # Complete the backpropagation of the coordinator

            for id in range(client_num):
                if id == 0:
                    a1 = client_data_index[id][0]
                    a2 = client_data_index[id][1]
                    td = client_list[id].calculate_gradient(dH_a[a1:a2], dH_p[a1:a2], dH_n[a1:a2])
                    for i, param in enumerate(net_glob_client.parameters()):
                        param.grad = td[i]
                else:
                    a1 = client_data_index[id][0]
                    a2 = client_data_index[id][1]
                    td = client_list[id].calculate_gradient(dH_a[a1:a2], dH_p[a1:a2], dH_n[a1:a2])
                    for i, param in enumerate(net_glob_client.parameters()):
                        param.grad = param.grad + td[i]
            # Simulate the cumulative model gradient at the aggregator

            optimizer_client.step()
            optimizer_client.zero_grad()
            # Complete the remaining backpropagation

            for id in range(client_num):
                client_list[id].updata_net_w(net_glob_client.state_dict())

            # Synchronize participant models

        s5 = time.time()
        print("train time: ", s5 - s1, "    loss:", loss.item())

        # test model
        if epoch % 1 == 0:
            net_glob_client.eval()
            net_glob_server.eval()
            with torch.no_grad():
                s6 = time.time()
                vali_node_list, _, vali_d2vec_list, vali_speed_list = dataload.load(load_part='test')
                embeddings = test_method.compute_embedding(roadnet=roadnet, net_c=net_glob_client,
                                                               net_s=net_glob_server,
                                                               test_traj=list(vali_node_list),
                                                               test_time=list(vali_d2vec_list),
                                                               test_speed=list(vali_speed_list),
                                                               test_batch=test_batch)
                acc = test_method.test_model(embeddings, isvali=False)
                ss, NMI, AMI, ARS, FMS = test_method.test_cluster(embeddings, k, "test")
                print("test time: ", time.time() - s6, '  epoch:', epoch)
                print(acc[0], acc[1], acc[2])
                print(NMI, AMI, ARS, FMS)
                print("")
                
                # save model
                save_model_client = 'Models/Pre_Models/{}/{}/participants/participant_checkpoint.pkl'.format(dataset, distance_type)
                torch.save(net_glob_client.state_dict(), save_model_client)


                save_model_server = 'Models/Pre_Models/{}/{}/coordinator/coordinator_checkpotin.pkl'.format(dataset, distance_type)
                torch.save(net_glob_server.state_dict(), save_model_server)

                if NMI > best_NMI:
                    best_NMI = NMI
                    best_NMI_epoch = epoch

                    best_NMI_client = 'Models/Pre_Models/{}/{}/participants/participant_NMI_BEST.pkl'.format(dataset, distance_type)
                    torch.save(net_glob_client.state_dict(), best_NMI_client)
                    best_NMI_server = 'Models/Pre_Models/{}/{}/coordinator/coordinator_NMI_BEST.pkl'.format(dataset, distance_type)
                    torch.save(net_glob_server.state_dict(), best_NMI_server)

                if acc[0] > best_HR10:
                    best_HR10 = acc[0]
                    best_HR10_epoch = epoch

                    best_HR10_client = 'Models/Pre_Models/{}/{}/participants/participant_HR10_BEST.pkl'.format(dataset, distance_type)
                    torch.save(net_glob_client.state_dict(), best_HR10_client)
                    best_HR10_server = 'Models/Pre_Models/{}/{}/coordinator/coordinator_HR10_BEST.pkl'.format(dataset, distance_type)
                    torch.save(net_glob_server.state_dict(), best_HR10_server)

                if epoch - best_NMI_epoch >= early_stop and epoch - best_HR10_epoch >= early_stop:
                    break
                

    print("the best clustring epoch is ", best_NMI_epoch, "the best NMI is ", best_NMI)
    print("the best similarity epoch is ", best_HR10_epoch, "the best HR10 is ", best_HR10)

    print("=======================================end pre train=======================================")

    return best_NMI_client, best_NMI_server


def test(load_client_model=None, load_server_model=None):
    if load_client_model != None and load_server_model != None:
        dataload = dataloaderUtils.DataLoader()
        net_glob_client = FMDTC_client(embedding_size, device,1,1,1)
        net_glob_server = FMDTC_server(cluster_size, embedding_size, k, device, nhead, num_layers, dropout,
                                       dim_feedforward,f_num=3)
        net_glob_client = net_glob_client.double().to(device)
        net_glob_server = net_glob_server.double().to(device)
        net_glob_client.load_state_dict(torch.load(load_client_model, map_location=device))
        net_glob_server.load_state_dict(torch.load(load_server_model, map_location=device))
        roadnet = dataloaderUtils.load_netowrk(dataset)
        net_glob_client = net_glob_client.to(device)
        net_glob_server = net_glob_server.to(device)
        net_glob_client.eval()
        net_glob_server.eval()
        with torch.no_grad():
            s6 = time.time()
            vali_node_list, _, vali_d2vec_list, vali_speed_list = dataload.load(load_part='test')
            embeddings = test_method.compute_embedding(roadnet=roadnet, net_c=net_glob_client,
                                                           net_s=net_glob_server,
                                                           test_traj=list(vali_node_list),
                                                           test_time=list(vali_d2vec_list),
                                                           test_speed=list(vali_speed_list),
                                                           test_batch=test_batch)
            acc = test_method.test_model(embeddings, isvali=False)
            s7 = time.time()
            print("test time: ", s7 - s6)
            print("similarity acc:", acc[0], acc[1], acc[2])
            NMI = []
            ss = []
            for i in range(0, 20):
                s_score, NMI_score, AMI, ARS, FMS = test_method.test_cluster(embeddings, k,"test")
                NMI.append(NMI_score)
                print(s_score, NMI_score)
            print("NMI_std", np.std(NMI))
            print("NMI_mean", np.mean(NMI))

