import numpy as np
import os
from utils import *


if __name__=="__main__":
    dataset_name = "BlogCatalog"
    split_dataset(dataset_name,ratio=0.7)
    #split_dataset(dataset_name,ratio=0.5)

    #gen_train_test_data(dataset_name,ratio=0.7)
    #random_gen_train_test_data(dataset_name,ratio=0.7)


