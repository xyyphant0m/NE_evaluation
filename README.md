# NE_evaluation
It's a self-use network embedding evaluation program

## Usage:

    evaluation.py [-h] [--task [TASK [TASK ...]]] --datasets-path

                    [DATASETS_PATH [DATASETS_PATH ...]] --embeddings-path

                    [EMBEDDINGS_PATH [EMBEDDINGS_PATH ...]] --seed SEED

                    [--Np NP] [--sampling SAMPLING]

                    [--eval-metrics [EVAL_METRICS [EVAL_METRICS ...]]]

                    [--start-train-ratio START_TRAIN_RATIO]

                    [--stop-train-ratio STOP_TRAIN_RATIO]

                    [--num-train-ratio NUM_TRAIN_RATIO] [--C C]

                    [--num-split NUM_SPLIT] [--num-workers NUM_WORKERS]


optional arguments:

    -h, --help            show this help message and exit

    --task [TASK [TASK ...]]

                        the evaluation task

    --datasets-path [DATASETS_PATH [DATASETS_PATH ...]]

                        input datasets path for datasets

    --embeddings-path [EMBEDDINGS_PATH [EMBEDDINGS_PATH ...]]

                        input embeddings path for embeddings

    --seed SEED         seed used for random number generator when randomly

                        split data into training/test set.

    --Np NP             number of Precision_k in network reconstruction or

                        link prediction

    --sampling SAMPLING sample the edge of graph in network reconstruction or

                        link prediction

    --eval-metrics [EVAL_METRICS [EVAL_METRICS ...]]

                        give the metrics in network reconstruction or link

                        prediction

    --start-train-ratio START_TRAIN_RATIO

                        the start value of the train ratio (inclusive).

    --stop-train-ratio STOP_TRAIN_RATIO

                        the end value of the train ratio (inclusive).

    --num-train-ratio NUM_TRAIN_RATIO

                        the number of train ratio choosed from [train-ratio-

                        start, train-ratio-end].

    --C C               inverse of regularization strength used in logistic

                        regression.

    --num-split NUM_SPLIT

                        The number of re-shuffling & splitting for each train

                        ratio.

    --num-workers NUM_WORKERS

                        Number of process in node classification





