import numpy as np
from scipy import sparse
from utils import (read_meta, read_probs, l2norm, knns2ordered_nbrs,
                   intdict2ndarray, Timer)


def read_ms1m(feat_path, label_path, knn_graph_path, feature_dim):
    with Timer('read meta and feature'):
        _, idx2lb = read_meta(label_path, verbose=False)
        inst_num = len(idx2lb)
        labels = intdict2ndarray(idx2lb)

        features = read_probs(feat_path, inst_num, feature_dim)
        features = l2norm(features)

    with Timer('read knn graph'):
        knns = np.load(knn_graph_path)['data']
        dists, knn_graph = knns2ordered_nbrs(knns, sort=True)
        sims = 1 - dists

    return features, labels, knn_graph, dists, sims


def read_ijb(feat_path, label_path, knn_graph_path):
    with Timer('read meta and feature'):
        features = np.load(feat_path)
        features = l2norm(features)
        labels = np.load(label_path)

        knn_graph = np.load(knn_graph_path)

        sims = list()
        for i in range(features.shape[0]):
            sims.append(features[knn_graph[i]] @ features[i])
        sims = np.stack(sims, axis=0)
        dists = 1 - sims

    return features, labels, knn_graph, dists, sims


def read_deepfashion(feat_path, label_path, knn_graph_path, feature_dim):
    with Timer('read meta and feature'):
        _, idx2lb = read_meta(label_path, verbose=False)
        inst_num = len(idx2lb)
        labels = intdict2ndarray(idx2lb)

        features = read_probs(feat_path, inst_num, feature_dim)
        features = l2norm(features)

    with Timer('read knn graph'):
        knns = np.load(knn_graph_path)
        dists, knn_graph = knns['dists'], knns['knns']
        sims = 1 - dists

    return features, labels, knn_graph, dists, sims


class PCENetDataset(object):
    def __init__(self, features, labels, knn_graph, dists, sims, k, scores):
        self.features = features
        self.labels = labels
        self.knn_graph = knn_graph
        self.dists = dists
        self.sims = sims

        ##############################################################################
        scores = scores[..., 1]
        num_nodes = len(self.features)
        knn_pair_flatten = np.stack(
            [np.repeat(np.expand_dims(np.arange(num_nodes), axis=1), k).reshape(-1, k), self.knn_graph],
            axis=-1).reshape(-1, 2)

        score_coo = sparse.coo_matrix((scores.reshape(-1), (knn_pair_flatten[:, 0], knn_pair_flatten[:, 1])),
                                      shape=(num_nodes, num_nodes), dtype=np.float32)
        score_csr = score_coo.tocsr()
        score_csr = (score_csr + score_csr.T) / 2
        scores_tmp = np.array(score_csr[knn_pair_flatten[:, 0], knn_pair_flatten[:, 1]]).reshape(-1, k)
        scores = scores_tmp

        self.density = np.sum((scores[:, 1:] * self.sims[:, 1:]), axis=1)

        density_diff_map = self.density[self.knn_graph[:]] > np.expand_dims(self.density[self.knn_graph[:]][:, 0], axis=1)

        sc_map = self.sims * scores
        sc_map[~density_diff_map] = 0

        E_d = np.zeros((len(self.features), k), dtype='bool')
        E_d[np.arange(self.features.shape[0]), np.argmax(sc_map, axis=1)] = True
        E_d[:, 0] = False

        row, col = np.where(E_d)
        self.E_d = np.stack([row, self.knn_graph[row, col]], axis=-1)
        self.E_d_labels = self.labels[self.E_d[:, 0]] == self.labels[self.E_d[:, 1]]

        E_s = self.sims * scores
        E_s[:, 0] = 0
        sim_threshold = np.mean(self.sims[:, 1:4])
        print('Similarity threshold: {}'.format(sim_threshold))
        E_s = E_s >= sim_threshold

        E_s[E_s & E_d] = False
        E_s = E_s & density_diff_map

        row, col = np.where(E_s)
        E_s = np.stack([row, self.knn_graph[row, col]], axis=-1)
        E_s = np.unique(np.sort(E_s, axis=1), axis=0)

        self.E_s = E_s

    def __getitem__(self, index):
        pair = self.E_d[index]
        label = self.E_d_labels[index]

        f1 = self.features[self.knn_graph[pair[0]]]
        f2 = self.features[self.knn_graph[pair[1]]]

        feature = np.concatenate([f1, f2], axis=0)
        adj = feature @ feature.T

        feature = feature.astype('float32')
        adj = adj.astype('float32')
        label = label.astype('int64')

        return (feature, adj), label, pair

    def __len__(self):
        return len(self.E_d)

    def len_nodes(self):
        return self.features.shape[0]

    def get_labels(self):
        return self.labels


class LCENetDataset(object):
    def __init__(self, dataset_name, feat_path, label_path, knn_graph_path, feature_dim, k):
        self.dataset_name = dataset_name
        self.feature_dim = feature_dim

        assert self.dataset_name in ['MS-Celeb-1M', 'IJB-B', 'DeepFashion']
        if self.dataset_name == 'MS-Celeb-1M':
            data = read_ms1m(feat_path=feat_path, label_path=label_path,
                             knn_graph_path=knn_graph_path, feature_dim=self.feature_dim)
        elif self.dataset_name == 'IJB-B':
            data = read_ijb(feat_path=feat_path, label_path=label_path, knn_graph_path=knn_graph_path)
        elif self.dataset_name == 'DeepFashion':
            data = read_deepfashion(feat_path=feat_path, label_path=label_path,
                                    knn_graph_path=knn_graph_path, feature_dim=self.feature_dim)

        self.features, self.labels, self.knn_graph, self.dists, self.sims = data

        # --------------------------------------------------
        self.knn_graph = self.knn_graph[:, :k]
        self.sims = self.sims[:, :k]
        self.dists = self.dists[:, :k]
        # --------------------------------------------------

    def __getitem__(self, index):
        features = self.features[self.knn_graph[index]]
        label = self.labels[index] == self.labels[self.knn_graph[index]]
        adj = features @ features.T

        features = features.astype('float32')
        adj = adj.astype('float32')
        label = label.astype('int64')

        return (features, adj), label

    def __len__(self):
        return len(self.features)

    def len_nodes(self):
        return self.features.shape[0]

    def get_labels(self):
        return self.labels
