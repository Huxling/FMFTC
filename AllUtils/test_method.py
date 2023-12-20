import os
import random
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import yaml
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import pairwise_distances
from AllUtils import clusterUtils
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score, adjusted_mutual_info_score



def compute_embedding(net_c, net_s, roadnet, test_traj, test_time, test_speed, test_batch):
    if len(test_traj) <= test_batch:
        smashed, length = net_c(roadnet, test_traj, test_time, test_speed, max(list(map(len, test_traj))))
        embedding = net_s(smashed, length, max(list(map(len, test_traj))), False)
        return embedding
    else:
        i = 0
        all_embedding = []
        while i < len(test_traj):
            smashed, length = net_c(roadnet, test_traj[i:i+test_batch], test_time[i:i+test_batch], test_speed[i:i+test_batch],
                                    max(list(map(len, test_traj[i:i+test_batch]))))
            embedding = net_s(smashed, length, max(list(map(len, test_traj[i:i+test_batch]))), False)
            all_embedding.append(embedding)
            i += test_batch

        all_embedding = torch.cat(all_embedding, 0)
        return all_embedding


def test_model(embedding_set, isvali=False):
    config = yaml.safe_load(open('config.yaml'))
    if isvali==True:
        input_dis_matrix = np.load(str(config["path_vali_truth"]))
    else:
        input_dis_matrix = np.load(str(config["path_test_truth"]))

    embedding_set = embedding_set.data.cpu().numpy()
    print(embedding_set.shape)

    embedding_dis_matrix = []
    for t in embedding_set:
        emb = np.repeat([t], repeats=len(embedding_set), axis=0)
        matrix = np.linalg.norm(emb-embedding_set, ord=2, axis=1)
        embedding_dis_matrix.append(matrix.tolist())

    l_recall_10 = 0
    l_recall_50 = 0
    l_recall_10_50 = 0

    f_num = 0

    for i in range(len(input_dis_matrix)):
        input_r = np.array(input_dis_matrix[i])
        one_index = []
        for idx, value in enumerate(input_r):
            if value != -1:
                one_index.append(idx)
        input_r = input_r[one_index]
        input_r = input_r[:5000]

        input_r50 = np.argsort(input_r)[1:51]
        input_r10 = input_r50[:10]

        embed_r = np.array(embedding_dis_matrix[i])
        embed_r = embed_r[one_index]
        embed_r = embed_r[:5000]

        embed_r50 = np.argsort(embed_r)[1:51]
        embed_r10 = embed_r50[:10]

        if len(one_index)>=51:
            f_num += 1
            l_recall_10 += len(list(set(input_r10).intersection(set(embed_r10))))
            l_recall_50 += len(list(set(input_r50).intersection(set(embed_r50))))
            l_recall_10_50 += len(list(set(input_r50).intersection(set(embed_r10))))

    recall_10 = float(l_recall_10) / (10 * f_num)
    recall_50 = float(l_recall_50) / (50 * f_num)
    recall_10_50 = float(l_recall_10_50) / (10 * f_num)

    return recall_10, recall_50, recall_10_50

def test_cluster(data, k, isvali="vali"):
    config = yaml.safe_load(open('config.yaml'))
    if isvali == "vali":
        y = np.load(str(config["cluster_vali_truth"]))
    else:
        y = np.load(str(config["cluster_test_truth"]))
    data = data.data.cpu()
    t1 = time.time()
    # y_pred = clusterUtils.sc_cluster(data, k)
    # y_pred = clusterUtils.dbscan(data)
    # y_pred, clu_loss, centroids = clusterUtils.run_kmeans(data, k)
    M, y_pred = clusterUtils.kMedoids(data, k)
    print("clustering time:", time.time() - t1)
    NMI_score = normalized_mutual_info_score(y, y_pred)
    s_score = silhouette_score(data, y_pred)
    return s_score, NMI_score, adjusted_mutual_info_score(y, y_pred), adjusted_rand_score(y, y_pred), fowlkes_mallows_score(y, y_pred)
