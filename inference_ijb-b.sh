python inference.py --k 120 \
                    --feature_dim 512 \
                    --lce_path ./weights/LCENet_CASIA.pth \
                    --pce_path ./weights/PCENet_CASIA.pth \
                    --dataset_name IJB-B \
                    --feat_path ./data/ijb-b/512.fea.npy \
                    --label_path ./data/ijb-b/512.labels.npy \
                    --knn_graph_path ./data/ijb-b/knn.graph.512.bf.npy