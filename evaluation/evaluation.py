import numpy as np
import scipy.sparse as sp
import argparse
import os
import collections
import random
from functools import reduce

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
import logging
from utils import *
import matplotlib
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

def construct_indicator(y_score, y):
    # rank the labels by the scores directly
    num_label = y.sum(axis=1, dtype=np.int32)
    # num_label = np.sum(y, axis=1, dtype=np.int)
    y_sort = np.fliplr(np.argsort(y_score, axis=1))
    #y_pred = np.zeros_like(y_score, dtype=np.int32)
    row, col = [], []
    for i in range(y_score.shape[0]):
        row += [i] * num_label[i, 0]
        col += y_sort[i, :num_label[i, 0]].tolist()
        #for j in range(num_label[i, 0]):
        #    y_pred[i, y_sort[i, j]] = 1
    y_pred = sp.csr_matrix(
            ([1] * len(row), (row, col)),
            shape=y.shape, dtype=np.bool_)
    return y_pred


def predict_cv(X, y, train_ratio=0.2, n_splits=10, random_state=0, C=1., num_workers=32):
    micro, macro = [], []
    shuffle = ShuffleSplit(n_splits=n_splits, test_size=1-train_ratio,
            random_state=random_state)
    for train_index, test_index in shuffle.split(X):
        #print(train_index.shape, test_index.shape)
        #assert len(set(train_index) & set(test_index)) == 0
        #assert len(train_index) + len(test_index) == X.shape[0]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    multi_class="ovr"),
                n_jobs=num_workers)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_pred = construct_indicator(y_score, y_test)
        mi = f1_score(y_test, y_pred, average="micro")
        ma = f1_score(y_test, y_pred, average="macro")
        #logger.info("micro f1 %f macro f1 %f", mi, ma)
        micro.append(mi)
        macro.append(ma)
    #logger.info("%d fold validation, training ratio %f", len(micro), train_ratio)
    #print("%d fold validation,train_ratio:%f\n",len(micro),train_ratio)
    #logger.info("Average micro %.2f, Average macro %.2f",np.mean(micro) * 100,np.mean(macro) * 100)
    return np.mean(micro)*100, np.mean(macro)*100

def sampling_edges(G, sampling, G_test=None):
    # when in link prediction, sampling means sample from G_test
    recon_flag = (G_test is None) # recon_flag = True (G_test==None) means it's in network reconstruction
    N = G.number_of_nodes()
    edges = []
    labels = []
    if sampling is None:
        logger.info("No sampling")
        for i in range(N):
            for j in range(i+1, N):
                if not recon_flag and G.has_edge(i, j):
                    continue
                edges.append([i, j])
                if recon_flag:
                    labels.append(1 if G.has_edge(i, j) else 0)
                else:
                    labels.append(1 if G_test.has_edge(i, j) else 0)
    else:
        edge_set = set()
        edges = []
        n = 0
        while True:
            i = np.random.randint(0, N-1)
            j = np.random.randint(i+1, N)
            if (i, j) not in edge_set:
                if not recon_flag and G.has_edge(i, j):
                    continue
                edge_set.add((i, j))
                edges.append([i, j])
                if recon_flag:
                    labels.append(1 if G.has_edge(i, j) else 0)
                else:
                    labels.append(1 if G_test.has_edge(i, j) else 0)
                n += 1
                if n % 1000000 == 0:
                    logger.info("sampling: {}/{}".format(n, sampling))
                if n >= sampling:
                    break
        pos = sum(labels)
        neg = len(labels)-pos
        logger.info("sampling: pos: %d, neg: %d",pos,neg)
    edges = np.array(edges).astype(int)
    labels = np.array(labels).astype(int)
    logger.info("edges.shape:%d,%d",edges.shape[0],edges.shape[1])
    logger.info("len of labels:%d",len(labels))
    return edges, labels

def reconstruction_link_prediction(edges, labels, embedding_filenames, eval_metrics, sampling, Np=1e6):
    res_precision_k = []
    res_auc = []
    Np = int(Np)
    for fn in embedding_filenames:
        emb = load_embeddings(fn) #support various file types
        logger.info("Embedding has shape %d, %d", emb.shape[0], emb.shape[1])
        if sampling is None:
            matrix_sim = emb.dot(emb.T)
            sim = matrix_sim[edges[:, 0], edges[:, 1]]
        else:
            #sim = dot_product(emb[edges[:, 0]], emb[edges[:, 1]])
            sim = batch_dot_product(emb, edges, batch_size=1e6)
            #sim = -euclidean_distance(emb[edges[:, 0]], emb[edges[:, 1]])
        ind = np.argsort(sim)[::-1] #sort descend
        assert len(ind) >= Np ,'Np too large'
        labels_ordered = labels[ind]
        logger.info(labels_ordered)
        if 'precision_k' in eval_metrics:
            positive = np.cumsum(labels_ordered[:Np+1])
            logger.info(positive)
            x = np.arange(Np+1)
            pk = (positive*1.0)/(x+1)

            fid,ax = plt.subplots()
            ax.semilogx(np.arange(Np+1),pk)
            ax.grid()
            plt.show()
            
            res_pk = pk[[10**i for i in range(0,7)]]
            res_precision_k.append(res_pk)
            logger.info("{},Precision_k:\t{}".format(os.path.basename(fn),pk))
        if 'AUC' in eval_metrics:
            rank = len(labels)-np.where(labels_ordered == 1)[0]
            M = len(rank)
            N = len(labels)-M
            auc = (np.sum(rank)-M*(M+1)/2)*1.0/M/N
            res_auc.append(auc)
            logger.info("{},AUC:\t{}".format(os.path.basename(fn),res_auc[-1]))
    return res_precision_k,res_auc

def node_classification(multi_label,embedding_filenames,args):
    seed = args.seed
    C = args.C
    start_train_ratio = args.start_train_ratio
    stop_train_ratio = args.stop_train_ratio
    num_train_ratio = args.num_train_ratio
    num_split = args.num_split
    num_workers = args.num_workers
    res_micro = []
    res_macro = []
    for fn in embedding_filenames:
        emb = load_embeddings(fn)
        logger.info("Network Embedding loaded!")
        logger.info("Embedding has shape %d, %d", emb.shape[0], emb.shape[1])
        num_label = multi_label.sum(axis=1, dtype=np.int32)
        idx = np.argwhere(num_label == 0)
        logger.info("%d instances with no label" % len(idx))
        #  if len(idx):
        #      embedding = embedding[label.getnnz(1)>0]
        #      label = label[label.getnnz(1)>0]
        #  logger.info("After deleting ...")
        train_ratios = np.linspace(start_train_ratio, stop_train_ratio, num_train_ratio)

        f1 = list()
        for tr in train_ratios:
            result = predict_cv(emb, multi_label, train_ratio=tr/100.,n_splits=num_split, C=C, random_state=seed,num_workers=num_workers)
            f1.append(result)
        micro, macro = zip(*f1)
        logger.info(os.path.basename(fn))
        logger.info(" ".join([str(x) for x in micro]))
        logger.info(" ".join([str(x) for x in macro]))
        res_micro.append(micro)
        res_macro.append(macro)
    return res_micro,res_macro

def evalution_task(dataset_names,embedding_filenames,args):
    np.random.seed(args.seed)
    sampling = args.sampling
    Np = args.Np
    eval_metrics = args.eval_metrics
    task = args.task
    if 'network_reconstruction' in task:
        logger.info("Begin network reconstruction")
        for filename in dataset_names:
            G = load_edgelist(filename)
            logger.info("Graph info: num of nodes-----{},num of edges-----{}".format(G.number_of_nodes(), G.number_of_edges()))
            edges,labels = sampling_edges(G,sampling)
            res = reconstruction_link_prediction(edges,labels,embedding_filenames,eval_metrics,args.sampling,Np)
    if 'link_prediction' in task:
        logger.info("Begin link prediction")
        for filename in dataset_names:
            G = load_edgelist(filename)
            logger.info("Graph info: num of nodes-----{},num of edges-----{}".format(G.number_of_nodes(), G.number_of_edges()))
            Path_name = os.path.dirname(filename)
            Data_name = os.path.basename(Path_name)
            dataset_train_name = os.path.join(Path_name,Data_name+"_0.7_train.edgelist")
            dataset_test_name = os.path.join(Path_name,Data_name+"_0.7_test.edgelist")
            G_train = load_edgelist(dataset_train_name)
            G_test = load_edgelist(dataset_test_name)
            logger.info("Graph_train info: num of nodes-----{},num of edges-----{}".format(G_train.number_of_nodes(), G_train.number_of_edges()))
            logger.info("Graph_test info: num of nodes-----{},num of edges-----{}".format(G_test.number_of_nodes(), G_test.number_of_edges()))
            edges,labels = sampling_edges(G_train,sampling,G_test=G_test)
            res = reconstruction_link_prediction(edges,labels,embedding_filenames,eval_metrics,sampling,Np)
    if 'node_classification' in task:
        logger.info("Begin node classification")
        for filename in dataset_names:
            logger.info(filename)
            dir_name = os.path.dirname(filename)
            data_name = os.path.basename(dir_name)
            label_name = os.path.join(dir_name,data_name+'.mat')
            multi_label = load_label(label_name)
            res = node_classification(multi_label,embedding_filenames,args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",nargs='*',type=str, default='node_classification',
            help='the evaluation task')
    parser.add_argument("--datasets-path",nargs='*',type=str,required=True,
            help="input datasets path for datasets")
    parser.add_argument("--embeddings-path",nargs='*',type=str,required=True,
            help="input embeddings path for embeddings")
    parser.add_argument("--seed", type=int, required=True,
            help="seed used for random number generator when randomly split data into training/test set.")
    parser.add_argument("--Np",type=float,default=1e6,
            help="number of Precision_k in network reconstruction or link prediction")
    parser.add_argument("--sampling",type=float,default=None,
            help="sample the edge of graph in network reconstruction or link prediction")
    parser.add_argument("--eval-metrics",nargs='*',
            help="give the metrics in network reconstruction or link prediction")
    parser.add_argument("--start-train-ratio", type=float, default=10,
            help="the start value of the train ratio (inclusive).")
    parser.add_argument("--stop-train-ratio", type=float, default=90,
            help="the end value of the train ratio (inclusive).")
    parser.add_argument("--num-train-ratio", type=int, default=9,
            help="the number of train ratio choosed from [train-ratio-start, train-ratio-end].")
    parser.add_argument("--C", type=float, default=1.0,
            help="inverse of regularization strength used in logistic regression.")
    parser.add_argument("--num-split", type=int, default=5,
            help="The number of re-shuffling & splitting for each train ratio.")
    parser.add_argument("--num-workers", type=int, default=32,
            help="Number of process in node classification")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    dataset_names = args.datasets_path
    embedding_filenames = args.embeddings_path
    logger.info(args)
    logger.info(dataset_names)
    logger.info(embedding_filenames)
    evalution_task(dataset_names,embedding_filenames,args)









