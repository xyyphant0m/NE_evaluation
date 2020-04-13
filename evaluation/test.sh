#!/bin/bash

set -x

python evaluation.py --datasets-path "../datasets/BlogCatalog/BlogCatalog.edgelist" \
                    --embeddings-path "../embeddings/BlogCatalog_alg1.emb" \
                    --task "network_reconstruction" "node_classification" --seed 0 --C 1 --Np 1e6 --eval-metrics "precision_k" "AUC" \
                    --num-split 5 --start-train-ratio 10 --stop-train-ratio 90 --num-train-ratio 9 --num-workers 32


