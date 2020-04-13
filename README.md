# NE_evaluation
It's a self-use network embedding evaluation program

## Usage:
evaluation.py [-h] [--task [TASK [TASK ...]]] --datasets-path<br/>
                     [DATASETS_PATH [DATASETS_PATH ...]] --embeddings-path<br/>
                     [EMBEDDINGS_PATH [EMBEDDINGS_PATH ...]] --seed SEED<br/>
                     [--Np NP] [--sampling SAMPLING]<br/>
                     [--eval-metrics [EVAL_METRICS [EVAL_METRICS ...]]]<br/>
                     [--start-train-ratio START_TRAIN_RATIO]<br/>
                     [--stop-train-ratio STOP_TRAIN_RATIO]<br/>
                     [--num-train-ratio NUM_TRAIN_RATIO] [--C C]<br/>
                     [--num-split NUM_SPLIT] [--num-workers NUM_WORKERS]<br/>


optional arguments:<br/>
  -h, --help            show this help message and exit<br/>
  --task [TASK [TASK ...]]<br/>
                        the evaluation task<br/>
  --datasets-path [DATASETS_PATH [DATASETS_PATH ...]]<br/>
                        input datasets path for datasets<br/>
  --embeddings-path [EMBEDDINGS_PATH [EMBEDDINGS_PATH ...]]<br/>
                        input embeddings path for embeddings<br/>
  --seed SEED           seed used for random number generator when randomly<br/>
                        split data into training/test set.<br/>
  --Np NP               number of Precision_k in network reconstruction or<br/>
                        link prediction<br/>
  --sampling SAMPLING   sample the edge of graph in network reconstruction or<br/>
                        link prediction<br/>
  --eval-metrics [EVAL_METRICS [EVAL_METRICS ...]]<br/>
                        give the metrics in network reconstruction or link<br/>
                        prediction<br/>
  --start-train-ratio START_TRAIN_RATIO<br/>
                        the start value of the train ratio (inclusive).<br/>
  --stop-train-ratio STOP_TRAIN_RATIO<br/>
                        the end value of the train ratio (inclusive).<br/>
  --num-train-ratio NUM_TRAIN_RATIO<br/>
                        the number of train ratio choosed from [train-ratio-<br/>
                        start, train-ratio-end].<br/>
  --C C                 inverse of regularization strength used in logistic<br/>
                        regression.<br/>
  --num-split NUM_SPLIT<br/>
                        The number of re-shuffling & splitting for each train<br/>
                        ratio.<br/>
  --num-workers NUM_WORKERS<br/>
                        Number of process in node classification<br/>





