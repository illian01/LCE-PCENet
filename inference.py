from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from scipy import sparse
import numpy as np
import time
import argparse

from datasets import PCENetDataset, LCENetDataset
from models import PCENet, LCENet
from metrics import pairwise, bcubed
from utils import labeling, build_symmetric_adj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size_lce', type=int, default=2048)
    parser.add_argument('--batch_size_pce', type=int, default=1024)

    parser.add_argument('--k', type=int, default=80)
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)

    parser.add_argument('--dataset_name', type=str, default='MS-Celeb-1M')
    parser.add_argument('--feat_path', type=str, default='./data/ms1m/features/part1_test.bin')
    parser.add_argument('--label_path', type=str, default='./data/ms1m/labels/part1_test.meta')
    parser.add_argument('--knn_graph_path', type=str, default='./data/ms1m/knns/part1_test/faiss_k_80.npz')

    parser.add_argument('--lce_path', type=str, default='./weights/LCENet_MS1M.pth')
    parser.add_argument('--pce_path', type=str, default='./weights/PCENet_MS1M_before.pth')

    args = parser.parse_args()

    device = torch.device('cuda')

    # LCENet inference
    model = LCENet(feature_dim=args.feature_dim, dim_feedforward=args.dim_feedforward, k=args.k,
                   num_layers=args.num_blocks, nhead=args.nhead)
    model.load_state_dict(torch.load(args.lce_path))
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    lce_dataset = LCENetDataset(dataset_name=args.dataset_name, feat_path=args.feat_path, label_path=args.label_path,
                                 knn_graph_path=args.knn_graph_path, feature_dim=args.feature_dim, k=args.k)
    lce_loader = DataLoader(lce_dataset, batch_size=args.batch_size_lce, num_workers=20,
                             shuffle=False, drop_last=False)

    model.eval()
    test_process = tqdm(lce_loader)
    scores = list()
    for step, data in enumerate(test_process):
        inputs, labels = (data[0][0].to(device), data[0][1].to(device)), data[1].to(device)

        output = model(inputs)
        scores.append(torch.softmax(output, dim=-1).detach().cpu().numpy())

    del lce_loader
    del model

    scores = np.concatenate(scores, axis=0)
    torch.cuda.empty_cache()

    # PCENet inference
    pce_dataset = PCENetDataset(features=lce_dataset.features, labels=lce_dataset.labels,
                                knn_graph=lce_dataset.knn_graph, dists=lce_dataset.dists, sims=lce_dataset.sims,
                                k=args.k, scores=scores)
    pce_loader = DataLoader(pce_dataset, batch_size=args.batch_size_pce, num_workers=20,
                             shuffle=False, drop_last=False)

    model = PCENet(feature_dim=args.feature_dim, k=args.k, nhid=1024, nclass=2)
    model.load_state_dict(torch.load(args.pce_path))

    model = torch.nn.DataParallel(model)

    model = model.to(device)

    start = time.time()

    model.eval()
    test_process = tqdm(pce_loader, )
    edges = list()
    test_acc = 0
    for step, data in enumerate(test_process):
        inputs, labels = (data[0][0].to(device), data[0][1].to(device)), data[1].to(device)
        pair = data[2].numpy()

        output = model(inputs)

        test_acc += (torch.argmax(output, dim=1) == labels).sum().item()
        output = torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy()

        edges.append(pair[output > 0.5])

    # Clustering
    edges = np.concatenate(edges, axis=0)
    edges = np.concatenate([edges, pce_dataset.E_s], axis=0)
    node_num = pce_dataset.len_nodes()

    adj_mat = sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(node_num, node_num))
    adj_mat = adj_mat.tocsr()
    adj_mat = build_symmetric_adj(adj_mat)

    pred = np.zeros(node_num) - 1

    start = time.time()
    labeling(pred, adj_mat)
    end = time.time()
    bfs_time = end - start

    gt = pce_dataset.get_labels()

    pairwise_fscore = pairwise(gt, pred)
    bcubed_fscore = bcubed(gt, pred)

    print('┌ Clustering result ─────────────────────────────┐')
    print('| Pairwise                                       |')
    print('| Precision   Recall   F-score                   |')
    print('| {:.4f}      {:.4f}   {:.4f}                    |'.format(pairwise_fscore[0], pairwise_fscore[1], pairwise_fscore[2]))
    print('├────────────────────────────────────────────────┤')
    print('| BCubed                                         |')
    print('| Precision   Recall   F-score                   |')
    print('| {:.4f}      {:.4f}   {:.4f}                    |'.format(bcubed_fscore[0], bcubed_fscore[1], bcubed_fscore[2]))
    print('└────────────────────────────────────────────────┘')
