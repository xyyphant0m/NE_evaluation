#!/bin/bash

set -x

python evaluation.py --datasets-path "../datasets/BlogCatalog/BlogCatalog.edgelist" \
    --embeddings-path "../embeddings/BlogCatalog_alg1.emb"\
    --task "network_reconstruction" --seed 0 --sim-methods  "dp" "cos" "euc" --eval-metrics "AUC" "precision_k"





