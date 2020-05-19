#!/bin/bash

set -x


python evaluation.py --datasets-path "../datasets/BlogCatalog/BlogCatalog.mat" \
    --embeddings-path  "../embeddings/BlogCatalog_alg1.emb"\
                    --task "node_classification" --seed 0 \
                    --C 1 --num-split 10 --start-train-ratio 10 --stop-train-ratio 90 --num-train-ratio 9 --num-workers 32


