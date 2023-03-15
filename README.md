# Local Connectivity-Based Density Estimation for Face Clustering

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

For DeepFashion dataset, we construct kNN graph using faiss.
- [Google Drive](https://drive.google.com/file/d/1ZfqX9gFoWxF2C9OGGY5yBMStd9Wggwbd/view?usp=sharing)

## Run
To test for each dataset, simply run shell scripts.

```
sh inference_{dataset_name}.sh
```

## Acknowledgement
Some codes are based on the publicly available codebase https://github.com/yl-1993/learn-to-cluster.

## Citation
```

```