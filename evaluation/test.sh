#!/bin/bash

set -x

#python evaluation.py --datasets-path "../../../3.NetWork_Embedding/datasets/BlogCatalog/BlogCatalog_0.7/BlogCatalog_0.7_train.edgelist" \
#                    --embeddings-path "../../../3.NetWork_Embedding/embeddings/BlogCatalog_0.7/BlogCatalog_0.7_randne.npy" \
#                    --task "link_prediction" --seed 0 --C 1 --Np 1e6 --eval-metrics "precision_k" "AUC" \
#                    --num-split 5 --start-train-ratio 10 --stop-train-ratio 90 --num-train-ratio 9 --num-workers 32

python evaluation.py --datasets-path "../datasets/BlogCatalog/BlogCatalog_0.7/BlogCatalog_0.7_train.edgelist" \
                    --embeddings-path "../../../3.NetWork_Embedding/embeddings/BlogCatalog_0.7/BlogCatalog_0.7_randne.npy" \
                    --task "link_prediction" --seed 0 --C 1 --Np 1e6 --eval-metrics "precision_k" "AUC" \
                    --num-split 5 --start-train-ratio 10 --stop-train-ratio 90 --num-train-ratio 9 --num-workers 32


