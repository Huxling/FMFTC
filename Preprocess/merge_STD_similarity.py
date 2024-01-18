import time

import numpy as np
import random


import yaml
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
setup_seed(2000)

config = yaml.safe_load(open('config.yaml'))
dataset = str(config["dataset"])

def merge_std_dis(valiortest = None):
    s = np.load('../ground_truth/{}/{}/{}_spatial_distance.npy'.format(str(config["dataset"]), str(config["distance_type"]), valiortest))
    t = np.load('../ground_truth/{}/{}/{}_temporal_distance.npy'.format(str(config["dataset"]), str(config["distance_type"]), valiortest))
    d = np.load('../ground_truth/{}/{}/{}_speed_distance.npy'.format(str(config["dataset"]), str(config["distance_type"]), valiortest))
    print(s.shape)

    coe = 1
    dcoe = 2
    stcoe = 4
    if str(config["distance_type"]) == "TP":
        coe = 8
    elif str(config["distance_type"]) == "DITA":
        coe = 32
    elif str(config["distance_type"]) == "LCRS":
        coe = 4
    elif str(config["distance_type"]) == "NetERP":
        coe = 8

    unreach = {}
    for i, dis in enumerate(s):
        tmp = []
        for j, und in enumerate(dis):
            if und == -1:
                tmp.append(j)
        if len(tmp)>0:
            unreach[i] = tmp
    s = (coe*stcoe)*s /np.max(s)
    t = (coe*stcoe)*t /np.max(t)
    d = (coe*dcoe)*d /np.max(d)

    std = (s+t+d)/10
    for i in unreach.keys():
        std[i][unreach[i]]=-1

    if valiortest == 'vali':
        np.save(str(config["path_vali_truth"]), std)
    elif valiortest == 'test':
        np.save(str(config["path_test_truth"]), std)
    elif valiortest == 'train':
        np.save(str(config["path_train_truth"]), std)

    print("complete: merge_std_distance")

def true_sc_cluster():

    config = yaml.safe_load(open('config.yaml'))
    test_dis_matrix = np.load(str(config["path_test_truth"]))
    k = int(config["cluster"]["k"])

    max1 = np.max(test_dis_matrix)
    test_dis_matrix[test_dis_matrix == -1] = max1
    affinity_matrix = (max1 - test_dis_matrix)

    print(affinity_matrix)
    sc_clustering = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=33).fit(affinity_matrix)
    y = sc_clustering.labels_

    np.save(str(config["cluster_test_truth"]), y)

def true_dbscan_cluster(valiortest = None):

    config = yaml.safe_load(open('config.yaml'))
    print(str(config["distance_type"]))
    if valiortest == 'vali':
        dis_matrix = np.load(str(config["path_vali_truth"]))
    elif valiortest == 'test':
        dis_matrix = np.load(str(config["path_test_truth"]))
    elif valiortest == 'train':
        dis_matrix = np.load(str(config["path_train_truth"]))
    else:
        return 0

    max1 = np.max(dis_matrix)
    dis_matrix[dis_matrix == -1] = max1

    dbscan = DBSCAN(eps=0.35, min_samples=5, metric="precomputed")
    labels = dbscan.fit_predict(dis_matrix)
    labels[labels == -1] = max(labels)+1
    # print(max(labels))
    if valiortest == 'vali':
        np.save(str(config["cluster_vali_truth"]), labels)
    elif valiortest == 'test':
        np.save(str(config["cluster_test_truth"]), labels)
    elif valiortest == 'train':
        np.save(str(config["cluster_train_truth"]), labels)
    print("complete: merge_std_distance")
def true_km_cluster(valiortest = None):
    t0 = time.time()
    config = yaml.safe_load(open('config.yaml'))
    print(str(config["distance_type"]))

    if valiortest == 'vali':
        dis_matrix = np.load(str(config["path_vali_truth"]))
    elif valiortest == 'test':
        dis_matrix = np.load(str(config["path_test_truth"]))
    elif valiortest == 'train':
        dis_matrix = np.load(str(config["path_train_truth"]))
    else:
        return 0

    corr = dis_matrix
    data_num = 5000
    Num = int(config["cluster"]["k"])
    max1 = np.max(corr)
    corr[corr == -1] = max1
    # corr = (max1 - corr) / max1
    corr = (max1 - corr)/(max1)

    bq = np.arange(data_num)
    u = np.arange(Num) 
    # u = np.random.choice(data_num,Num,replace=False)
    for n in range(1000): 
        cluster = [[v] for v in u]  
        for i, t in enumerate(cluster):
            bq[t[0]] = i
        others = np.array([v for v in range(data_num) if v not in u])  
        temp = corr[:, u]
        temp = temp[others, :] 
        inds = temp.argmax(axis=1)  
        new_u = []
        for i in range(Num):  
            ind = np.where(inds == i)[0]  
            points = others[ind] 
            cluster[i] = cluster[i] + points.tolist()  
            bq[points] = i
            temp = corr[cluster[i], :]
            temp = temp[:, cluster[i]]  
            ind_ = temp.sum(axis=0).argmax() 
            ind_new_center = cluster[i][ind_]  
            new_u.append(ind_new_center) 
        new_u = np.asarray(new_u, dtype=np.int32)
        if (new_u == u).sum() == Num: 
            break
        print(n, (new_u == u).sum(), time.time() - t0)
        u = new_u.copy()  


    if valiortest == 'vali':
        np.save(str(config["cluster_vali_truth"]), bq)
    elif valiortest == 'test':
        np.save(str(config["cluster_test_truth"]), bq)
    print("complete: merge_std_distance")
    
if __name__ == "__main__":

    merge_std_dis(valiortest='test')
    true_km_cluster(valiortest='test')
    merge_std_dis(valiortest='vali')
    true_km_cluster(valiortest='vali')

