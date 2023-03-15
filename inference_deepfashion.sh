python inference.py --k 8 \
                    --feature_dim 256 \
                    --lce_path ./weights/LCENet_DeepFashion.pth \
                    --pce_path ./weights/PCENet_DeepFashion.pth \
                    --dataset_name DeepFashion \
                    --feat_path ./data/deepfashion/features/deepfashion_test.bin \
                    --label_path ./data/deepfashion/labels/deepfashion_test.meta \
                    --knn_graph_path ./data/deepfashion/knns/deepfashion_test/deepfashion_k40.npz