#!/bin/bash

set -x

python evaluation.py --datasets-path "../datasets2/BlogCatalog/BlogCatalog_0.7/BlogCatalog_0.7_train.edgelist" \
    --embeddings-path  "../embeddings2/BlogCatalog_0.7/BlogCatalog_0.7_alg1.emb"\
    --task "link_prediction" --seed 0 --sim-methods  "dp" "cos" "euc" "avg" "had" "l1" "l2" --eval-metrics "AUC"




