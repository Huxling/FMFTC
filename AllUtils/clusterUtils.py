import pickle
import time
import yaml
import faiss
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import torch
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import SpectralClustering
config = yaml.safe_load(open('config.yaml'))
from sklearn.cluster import DBSCAN

def run_kmeans(x, k, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        (list: ids of data in each cluster, float: loss value)
    """
    # pca = PCA(n_components=6)
    # pca.fit(x)
    # x = pca.transform(x)
    # x = x.astype(np.float32)
    n_data, d = x.shape
    # config = yaml.safe_load(open('config.yaml'))
    # device = int(config["cuda"])
    niter = 20
    seed = np.random.randint(2000)
    x = x.to(torch.float32)
    # print("data", x)
    init_centroids = x[:k]
    # print(init_centroids)
    kmeans = faiss.Kmeans(d, k, niter=niter, verbose=verbose, seed=seed, gpu=0)

    kmeans.train(x, init_centroids=np.array(init_centroids))
    D, I = kmeans.index.search(x, 1)

    stats = kmeans.iteration_stats
    losses = np.array([
        stats[i] for i in range(len(stats))
    ])

    centroids = kmeans.centroids

    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1], centroids

class Kmeans(object):
    def __init__(self, k):
        self.k = k
        self.data_lists = [[] for i in range(self.k)]

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # cluster the data
        I, loss, centroids = run_kmeans(data, self.k, verbose)
        for i in range(len(data)):
            self.data_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return I, loss, centroids

def pseudoLabels(k, data):
    s3 = time.time()
    # kmeans = Kmeans(k)
    # I, loss, centroids = kmeans.cluster(data)
    M, I = kMedoids(data, k)
    # I = dbscan(data)
    # I = sc_cluster(data, k)
    print("===cluster time: ", time.time() - s3)
    pickle.dump(I, open(str(config["pseudo_labels"]), 'wb'))
    # return kmeans.data_lists, I, loss, centroids

def dbscan(data):
    dbscan = DBSCAN(eps=0.35, min_samples=5)
    labels = dbscan.fit_predict(data)
    labels[labels == -1] = max(labels)+1
    return labels
def sc_cluster(data,k):
    test_dis_matrix = cdist(data, data, 'euclidean')

    max1 = np.max(test_dis_matrix)
    test_dis_matrix[test_dis_matrix == -1] = max1
    affinity_matrix = (max1 - test_dis_matrix)

    print(affinity_matrix)
    sc_clustering = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=33).fit(affinity_matrix)
    y = sc_clustering.labels_

    return y

def kMedoids(data, k, tmax=100):
    # determine dimensions of distance matrix D

    D = pairwise_distances(data, metric='euclidean')

    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs,cs = np.where(D==0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r,c in zip(rs,cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    # np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
    Y = [0 for i in range(len(D))]
    # return results
    for label in C:
        for id in C[label]:
            Y[id] = label
    return M, Y

