# Local Connectivity-Based Density Estimation for Face Clustering
This repo contains an official implementation for CVPR'23 paper "Local Connectivity-Based Density Estimation for Face Clustering".

![](https://raw.githubusercontent.com/illian01/LCE-PCENet/main/assets/overview.jpg)

## Introduction
The
proposed clustering method employs density-based clustering, which maintains edges that have higher density. For this purpose, we propose a reliable density estimation algorithm based on local connectivity between K nearest neighbors (KNN). We effectively exclude negative pairs from the KNN graph based on the reliable density while maintaining sufficient positive pairs. Furthermore, we develop a pairwise connectivity estimation network to predict the connectivity of the selected edges.

## Requirements
- Python = 3.8.5
- Pytorch = 1.10.2
```
conda install pytorch==1.10.2 cudatoolkit=11.3 -c pytorch
pip install -r requriements.txt
```

## Dataset
The data directory is constructed as follows:
```
data
  ├── ms1m
  |    ├── features
  |    |    ├── part1_test.bin
  |    |    ├── ...
  |    |    └── part9.test.bin
  |    ├── labels
  |    |    ├── part1_test.meta
  |    |    ├── ...
  |    |    └── part9.test.meta
  |    └── knns
  |         ├── part1_test
  |         |    └── faiss_k_80.npz
  |         ├── ...
  |         └── part9_test
  |              └── faiss_k_80.npz
  ├── deepfashion
  |    ├── features
  |    |    └── deepfashion_test.bin
  |    ├── labels
  |    |    └── deepfashion_test.meta
  |    └── knns
  |         └── deepfashion_test
  |              └── deepfashion_k40.npz
  └── ijb-b
       ├── 512.fea.npy
       ├── 512.labels.npy
       ├── knn.graph.512.bf.npy
       └── ...
```

We have used the data from following repositories. 
- [MS-Celeb-1M, DeepFashion] https://github.com/yl-1993/learn-to-cluster
- [CASIA/IJB-B] https://github.com/Zhongdao/gcn_clustering

For DeepFashion dataset, we construct kNN graph using [faiss](https://github.com/facebookresearch/faiss).
- [Google Drive](https://drive.google.com/file/d/1ZfqX9gFoWxF2C9OGGY5yBMStd9Wggwbd/view?usp=sharing)

## Run
To test for each dataset, simply run shell scripts.

```
sh inference_{dataset_name}.sh
```

## Results on MS-Celeb-1M

1, 3, 5, 7, 9 mean different subset of the clustering benchmark. Detailed settings are on our paper.

### Pairwise F-Score
| Methods | 1 | 3 | 5 | 7 | 9 |
| ---------------- |:-:|:-:|:-:|:-:|:-:|
| [CDP](https://github.com/XiaohangZhan/cdp) |75.02|70.75|69.51|68.62|68.06|
| [L-GCN](https://github.com/Zhongdao/gcn_clustering) |78.68|75.83|74.29|73.71|72.99|
| [LTC](https://github.com/yl-1993/learn-to-cluster) |85.66|82.41|80.32|78.98|77.87|
| [GCN(V+E)](https://github.com/yl-1993/learn-to-cluster) |87.93|84.04|82.10|80.45|79.30|
| [Clusformer](https://github.com/uark-cviu/Intraformer/tree/master/Clusformer) |88.20|84.60|82.79|81.03|79.91|
| [STAR-FC](https://github.com/sstzal/STAR-FC) |91.97|88.28|86.17|84.70|83.46 |
| Pair-Cls |90.67|86.91|85.06|83.51|82.41|
| [Ada-NETS](https://github.com/damo-cv/Ada-NETS) |92.79|89.33|87.50|85.40|83.99|
| [Chen *et al.*](https://github.com/echoanran/On-Mitigating-Hard-Clusters) |93.22|90.51|89.09|87.93|86.94|
| Ours |94.64|91.90|90.27|88.69|87.35|

### BCubed F-Score
| Methods | 1 | 3 | 5 | 7 | 9 |
| ---------------- |:-:|:-:|:-:|:-:|:-:|
| [CDP](https://github.com/XiaohangZhan/cdp) |78.70|75.82|74.58|73.62|72.92|
| [L-GCN](https://github.com/Zhongdao/gcn_clustering) |84.37|81.61|80.11|79.33|78.60|
| [LTC](https://github.com/yl-1993/learn-to-cluster) |85.52|83.01|81.10|79.84|78.86|
| [GCN(V+E)](https://github.com/yl-1993/learn-to-cluster) |86.09|82.84|81.24|80.09|79.25|
| [Clusformer](https://github.com/uark-cviu/Intraformer/tree/master/Clusformer) |87.17|84.05|82.30|80.51|79.95|
| [STAR-FC](https://github.com/sstzal/STAR-FC) |-|86.26|84.13|82.63|81.47|
| Pair-Cls |89.54|86.25|84.55|83.49|82.40|
| [Ada-NETS](https://github.com/damo-cv/Ada-NETS) |91.40|87.98|86.03|84.48|83.28|
| [Chen *et al.*](https://github.com/echoanran/On-Mitigating-Hard-Clusters) |92.18|89.43|88.00|86.92|86.06|
| Ours |93.36|90.78|89.28|88.15|87.28|

## Results on IJB-B

$F_{512}, F_{1024}, F_{1845}$ mean different subset of the clustering benchmark. Detailed settings are on our paper.

### Pairwise F-Score
| Methods | $F_{512}$ | $F_{1024}$ | $F_{1845}$ |
| ---------------- |:-:|:-:|:-:|
| Pair-Cls |84.4|83.3|82.7|
| [Chen *et al.*](https://github.com/echoanran/On-Mitigating-Hard-Clusters) |80.8|73.2|59.1|
| Ours |93.0|92.7|90.8|

### BCubed F-Score
| Methods | $F_{512}$ | $F_{1024}$ | $F_{1845}$ |
| ---------------- |:-:|:-:|:-:|
| [L-GCN](https://github.com/Zhongdao/gcn_clustering) |83.3|83.3|81.4|
| DANet |83.4|83.3|82.8|
| [Chen *et al.*](https://github.com/echoanran/On-Mitigating-Hard-Clusters) |79.6|78.1|76.7|
| Ours |85.4|85.2|84.8|

## Results on DeepFashion

| Methods | Pairwise F-Score | BCubed F-Score |
| ---------------- |:-:|:-:|
| [CDP](https://github.com/XiaohangZhan/cdp) |28.28|57.83|
| [L-GCN](https://github.com/Zhongdao/gcn_clustering) |28.85|58.91|
| [LTC](https://github.com/yl-1993/learn-to-cluster) |29.14|59.11|
| [GCN(V+E)](https://github.com/yl-1993/learn-to-cluster) |38.47|60.06|
| Pair-Cls |37.67|62.17|
| [Ada-NETS](https://github.com/damo-cv/Ada-NETS) |39.30|61.05|
| [Chen *et al.*](https://github.com/echoanran/On-Mitigating-Hard-Clusters) |40.91|63.61|
| Ours |41.76|64.56|

## Acknowledgement
Some codes are based on the publicly available codebase https://github.com/yl-1993/learn-to-cluster.

## Citation
```

```