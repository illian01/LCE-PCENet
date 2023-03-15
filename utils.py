import time
import numpy as np
from collections import deque
from scipy import sparse


class Timer():
    def __init__(self, name='task', verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print('[Time] {} consumes {:.4f} s'.format(
                self.name,
                time.time() - self.start))
        return exc_type is None


def read_meta(fn_meta, start_pos=0, verbose=True):
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    if verbose:
        print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    return lb2idxs, idx2lb


def read_probs(path, inst_num, feat_dim, dtype=np.float32, verbose=False):
    assert (inst_num > 0 or inst_num == -1) and feat_dim > 0
    count = -1
    if inst_num > 0:
        count = inst_num * feat_dim
    probs = np.fromfile(path, dtype=dtype, count=count)
    if feat_dim > 1:
        probs = probs.reshape(inst_num, feat_dim)
    if verbose:
        print('[{}] shape: {}'.format(path, probs.shape))
    return probs


def l2norm(vec):
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


def knns2ordered_nbrs(knns, sort=True):
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :].astype(np.int32)
    dists = knns[:, 1, :]
    if sort:
        # sort dists from low to high
        nb_idx = np.argsort(dists, axis=1)
        idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
        dists = dists[idxs, nb_idx]
        nbrs = nbrs[idxs, nb_idx]
    return dists, nbrs


def intdict2ndarray(d, default_val=-1):
    arr = np.zeros(len(d)) + default_val
    for k, v in d.items():
        arr[k] = v
    return arr


def clusters2labels(clusters):
    idx2lb = {}
    for lb, cluster in enumerate(clusters):
        for v in cluster:
            idx2lb[v] = lb
    return idx2lb


def bfs(pred, adj_mat, visited, start, l):
    q = deque([start])
    visited[start] = True
    pred[start] = l

    while q:
        v = q.popleft()

        neighbors = adj_mat[v].indices
        for n in neighbors:
            if not visited[n]:
                q.append(n)
                visited[n] = True
                pred[n] = l


def labeling(pred, adj_mat):
    l = 0
    visited = np.zeros(len(pred), dtype=bool)

    for idx, p in enumerate(pred):
        if p == -1:
            bfs(pred, adj_mat, visited, idx, l)
            l += 1


def build_symmetric_adj(adj, self_loop=False):
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if self_loop:
        adj = adj + sparse.eye(adj.shape[0])
    return adj