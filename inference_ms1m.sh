python inference.py --k 80 \
                    --feature_dim 256 \
                    --lce_path ./weights/LCENet_MS1M.pth \
                    --pce_path ./weights/PCENet_MS1M.pth \
                    --dataset_name MS-Celeb-1M \
                    --feat_path ./data/ms1m/features/part1_test.bin \
                    --label_path ./data/ms1m/labels/part1_test.meta \
                    --knn_graph_path ./data/ms1m/knns/part1_test/faiss_k_80.npz