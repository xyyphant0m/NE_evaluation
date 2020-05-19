#!/usr/bin/env python3
#
# A python 3 script that facilitates running experiments.
# It does NOT contain implementation of any network embedding algorithm.
# Rather, it evaluates the performance of several embedding algorithms,
# in terms of Link Prediction and Node Classification.
# It probably won't be helpful to you.
# (Arguments / hyper-parameters are hard-coded :-b I'm sorry.)
#
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pickle
import subprocess
import sys

import networkx as nx
import numpy as np
import scipy.io
import scipy.sparse
import sklearn.linear_model
import sklearn.metrics
import sklearn.multiclass
from sklearn.preprocessing import normalize
from sklearn.metrics import average_precision_score

if sys.version[0] == '3':
    import urllib.request as urllib
else:
    import urllib


class NetData(object):
    def __init__(self, work_dir, name):
        if not os.path.exists(work_dir):
            os.mkdir(work_dir)
        if name in ["cora", "citeseer"]:
            self.network, self.label = NetData._load_linqs(work_dir, name)
        elif name in ["blogcatalog", "flickr", "youtube", "PPI", "POS"]:
            mat_file = os.path.join(work_dir, name + '.mat')
            network, label = NetData._load_tang09_mat(mat_file)
            self.network = NetData._mat_to_net(network)
            self.label = label.toarray().astype(np.bool)
        else:
            assert False

    @staticmethod
    def _load_linqs(work_dir, name):
        tgz_path = os.path.join(work_dir, name + ".tgz")
        url = "https://linqs-data.soe.ucsc.edu/public/lbc/%s.tgz" % name
        if not os.path.isfile(tgz_path):
            urllib.urlretrieve(url, tgz_path)
        subprocess.check_call(["tar", "xvzf", name + ".tgz"],
                              cwd=work_dir)
        if name == 'cora':
            n, num_wrds, num_cls = 2708, 1433, 7
        elif name == 'citeseer':
            n, num_wrds, num_cls = 3312, 3703, 6
        else:
            assert False
        pid2nid = {}
        cln2cid = {}
        y = np.zeros((n, num_cls), dtype=np.bool)
        with open(os.path.join(work_dir, name, name + '.content')) as fin:
            cnt = 0
            for line in fin:
                cnt += 1
                line = line.split()
                assert len(line) == 2 + num_wrds
                pid, cln = line[0], line[-1]
                assert pid not in pid2nid
                pid2nid[pid] = len(pid2nid)
                if cln not in cln2cid:
                    cln2cid[cln] = len(cln2cid)
                y[pid2nid[pid], cln2cid[cln]] = True
            assert cnt == n
            assert len(cln2cid) == num_cls
        g = nx.Graph()
        with open(os.path.join(work_dir, name, name + '.cites')) as fin:
            for line in fin:
                u, v = line.split()
                if (u in pid2nid) and (v in pid2nid):
                    u, v = pid2nid[u], pid2nid[v]
                    g.add_edge(u, v)
        '''
        nx.write_edgelist(g,"./cora/cora.edgelist",data=False)
        #nx.write_edgelist(g,"./citeseer/citeseer.edgelist",data=False)
        node_number = g.number_of_nodes()
        A = scipy.sparse.lil_matrix((node_number,node_number))
        for e in g.edges():
            A[e[0],e[1]]=1
            A[e[1],e[0]]=1
        A = scipy.sparse.csr_matrix(A)
        label = scipy.sparse.csr_matrix(y)
        scipy.io.savemat("./cora/cora.mat",{"network":A,"group":label})
        #scipy.io.savemat("./citeseer/citeseer.mat",{"network":A,"group":label})
        '''
        return g, y

    @staticmethod
    def _load_tang09_mat(mat_data_path):
        """
        Load a dataset in MATLAB format.
        Data source: http://leitang.net/social_dimension.html.
        """
        _, data_file = os.path.split(mat_data_path)
        assert data_file in ["blogcatalog.mat", "flickr.mat",
                             "youtube.mat", "PPI.mat", "POS.mat"]
        if not os.path.isfile(mat_data_path):
            if data_file in ["PPI.mat", "POS.mat"]:
                origin = "http://snap.stanford.edu/node2vec/"
            else:
                origin = "http://leitang.net/code/social-dimension/data/"
            if data_file == "PPI.mat":
                data_file = "Homo_sapiens.mat"
            url = origin + data_file
            urllib.urlretrieve(url, mat_data_path)
        mat = scipy.io.loadmat(mat_data_path)
        return mat["network"], mat["group"]

    @staticmethod
    def _mat_to_net(network_mat):
        g = nx.Graph()
        g.add_nodes_from(range(network_mat.shape[0]))
        coo = scipy.sparse.coo_matrix(network_mat)
        for u, v, _ in zip(coo.row, coo.col, coo.data):
            g.add_edge(u, v)
        assert g.number_of_nodes() == network_mat.shape[0]
        return g


class MultiLabelEval(object):
    @staticmethod
    def eval(train_x, train_y, test_x, test_y):
        classifier = sklearn.multiclass.OneVsRestClassifier(
            sklearn.linear_model.LogisticRegression(), n_jobs=-1)
        classifier.fit(train_x, train_y)
        score = classifier.predict_proba(test_x)
        return MultiLabelEval._compute_all_errors(test_y, score)

    @staticmethod
    def _preserve_k_top(score, k, axis):
        assert score.ndim == 2
        assert k.shape == (score.shape[1 - axis],)
        index = np.argsort(-score, axis=axis)
        pred = np.zeros_like(score, np.int)
        for i in range(score.shape[1 - axis]):
            if axis == 0:
                pred[index[:k[i], i], i] = 1
            else:
                pred[i, index[i, :k[i]]] = 1
        return pred

    @staticmethod
    def _compute_tang09_error(score, y, label_wise=False):
        """
        Translated from a MATLAB script provided by Tang & Liu. See:
            Relational learning via latent social dimensions. KDD '09.
        """
        assert score.ndim == 2
        assert score.shape == y.shape
        assert y.dtype in (np.int, np.bool)
        if y.dtype == np.bool:
            y = y.astype(np.int)
        index = (np.sum(y, axis=1) > 0)  # remove samples with no labels
        y = y[index]
        score = score[index]
        if label_wise:
            # Assuming the number of samples per label is known,
            # preserve only top-scored samples for each label.
            pred = MultiLabelEval._preserve_k_top(score, np.sum(y, 0), 0)
        else:
            # Assuming the number of labels per sample is known,
            # preserve only top-scored labels for each sample.
            pred = MultiLabelEval._preserve_k_top(score, np.sum(y, 1), 1)
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = sklearn.metrics.f1_score(y, pred,
                                                        average=average)
        results['accuracy'] = sklearn.metrics.accuracy_score(y, pred)
        return results

    @staticmethod
    def _compute_all_errors(y, score):
        perf = {}
        tang09_metrics = MultiLabelEval._compute_tang09_error(score, y)
        perf['macro_f1'] = tang09_metrics['macro']
        perf['micro_f1'] = tang09_metrics['micro']
        perf['accuracy'] = tang09_metrics['accuracy']
        perf['coverage_error'] = (sklearn.metrics.coverage_error(y, score))
        perf['label_ranking_average_precision_score'] = (
            sklearn.metrics.label_ranking_average_precision_score(y, score))
        perf['label_ranking_loss'] = (
            sklearn.metrics.label_ranking_loss(y, score))
        return perf


class BaseLinkPredictor(object):
    def __init__(self, feats, trunc):
        self.feats = feats
        self.trunc = trunc

    def train(self, tp_list, fp_list):
        assert len(tp_list[0]) == 2 and len(fp_list[0]) == 2
        assert self.trunc.number_of_nodes() > 1

    def get_sim(self, u, v):
        assert self.trunc.has_node(u) and self.trunc.has_node(v)
        return np.dot(self.feats[u], self.feats[v])

class BaseLinkPredictorCosine(BaseLinkPredictor):
    def get_sim(self, u, v):
        assert self.trunc.has_node(u) and self.trunc.has_node(v)
        d = self.feats[u].shape[0]
        fu = normalize(self.feats[u].reshape(1,d),norm="l2",axis=1).reshape(-1)
        fv = normalize(self.feats[v].reshape(1,d),norm="l2",axis=1).reshape(-1)
        return np.dot(fu, fv)

class BaseLinkPredictorEuclidean(BaseLinkPredictor):
    def get_sim(self, u, v):
        assert self.trunc.has_node(u) and self.trunc.has_node(v)
        fu = self.feats[u]
        fv = self.feats[v]
        return -np.linalg.norm(fu-fv)

class CommNeibLP(BaseLinkPredictor):
    def get_sim(self, u, v):
        a = set(self.trunc.neighbors(u))
        b = set(self.trunc.neighbors(v))
        return len(a.intersection(b))

class JaccardLP(BaseLinkPredictor):
    def get_sim(self, u, v):
        a = set(self.trunc.neighbors(u))
        b = set(self.trunc.neighbors(v))
        return len(a.intersection(b)) / (1.0 + len(a.union(b)))

class AdamicLP(BaseLinkPredictor):
    def get_sim(self, u, v):
        a = set(self.trunc.neighbors(u))
        b = set(self.trunc.neighbors(v))
        s = 0.0
        for t in a.intersection(b):
            s += 1.0 / np.log(1.0 + len(set(self.trunc.neighbors(t))))
        return s

class PreferLP(BaseLinkPredictor):
    def get_sim(self, u, v):
        a = set(self.trunc.neighbors(u))
        b = set(self.trunc.neighbors(v))
        return len(a) * len(b)

class HadamardLP(BaseLinkPredictor):
    def __init__(self, feats, trunc):
        BaseLinkPredictor.__init__(self, feats, trunc)
        self.cls = sklearn.linear_model.LogisticRegression()

    def train(self, tp_list, fp_list):
        x = self.get_train_x(tp_list + fp_list)
        y = np.zeros(len(tp_list) + len(fp_list), dtype=np.int64)
        y[:len(tp_list)] = 1
        self.cls.fit(x, y)

    def get_train_x(self, edges):
        n = len(edges)
        d = self.feats.shape[1]
        x = np.zeros((n, d), dtype=np.float64)
        for i, e in enumerate(edges):
            x[i] = self.get_edge_feat(*e)
        return x

    def get_sim(self, u, v):
        d = self.feats.shape[1]
        x = np.zeros((1, d), dtype=np.float64)
        x[0] = self.get_edge_feat(u, v)
        #p = self.cls.predict(x)
        p = self.cls.predict_proba(x)[:,1]
        return p[0]

    def get_edge_feat(self, u, v):
        return self.feats[u] * self.feats[v]

class AverageLP(HadamardLP):
    def get_edge_feat(self, u, v):
        return (self.feats[u] + self.feats[v]) / 2.0

class L1LP(HadamardLP):
    def get_edge_feat(self, u, v):
        return np.abs(self.feats[u] - self.feats[v])

class L2LP(HadamardLP):
    def get_edge_feat(self, u, v):
        return (self.feats[u] - self.feats[v]) ** 2

class LinkPredEval(object):
    @staticmethod
    def hide_links(whole, hide_ratio=0.50):
        trunc = whole.copy()
        for u in range(whole.number_of_nodes()):
            assert trunc.degree(u) > 0
        edges = list(whole.edges())
        np.random.shuffle(edges)
        for (u, v) in edges:
            if u != v and trunc.degree(u) > 1 and trunc.degree(v) > 1:
                if np.random.rand() < hide_ratio:
                    trunc.remove_edge(u, v)
        for u in range(whole.number_of_nodes()):
            assert trunc.degree(u) > 0
        assert trunc.number_of_nodes() == whole.number_of_nodes()
        print("%d/%d edges kept in TRUNC" % (
            trunc.number_of_edges(), whole.number_of_edges()))
        return trunc

    @staticmethod
    def eval(feats, trunc, whole, work_dir, suffix=''):
        whole_edges = set(whole.edges())
        trunc_edges = set(trunc.edges())
        tp = whole_edges - trunc_edges
        predictors = [BaseLinkPredictor,BaseLinkPredictorCosine,BaseLinkPredictorEuclidean,
                      CommNeibLP, JaccardLP, AdamicLP, PreferLP,
                      HadamardLP, AverageLP, L1LP, L2LP]
        predictors = [p(feats, trunc) for p in predictors]
        train_tp = list(trunc.edges())
        
        pkl_file = os.path.join(work_dir, 'aucfp' + suffix + ".train")
        if os.path.exists(pkl_file):
            print("loading train_fp pkl_file...")
            with open(pkl_file, 'rb') as fin:
                train_fp = list(pickle.load(fin))
        else:
            train_fp = [LinkPredEval._sample_false(whole) for _ in range(len(train_tp))]
            fp = set(train_fp)
            with open(pkl_file, 'wb') as fout:
                pickle.dump(fp, fout)
        
        #train_fp = [LinkPredEval._sample_false(whole) for _ in range(len(train_tp))]
        for pred in predictors:
            pred.train(train_tp, train_fp)
        auc_scores,ap_scores = LinkPredEval._get_rocs(predictors, tp, whole, work_dir,
                                        suffix)
        return {
            'auc_roc_dot': auc_scores[0],
            'auc_roc_cosine':auc_scores[1],
            'auc_roc_euclidean':auc_scores[2],
            'auc_roc_common': auc_scores[3],
            'auc_roc_jaccard': auc_scores[4],
            'auc_roc_adamic': auc_scores[5],
            'auc_roc_prefer': auc_scores[6],
            'auc_roc_hadamard': auc_scores[7],
            'auc_roc_average': auc_scores[8],
            'auc_roc_L1': auc_scores[9],
            'auc_roc_L2': auc_scores[10]
        },{
            'ap_dot': ap_scores[0],
            'ap_cosine': ap_scores[1],
            'ap_euclidean': ap_scores[2],
            'ap_common': ap_scores[3],
            'ap_jaccard': ap_scores[4],
            'ap_adamic': ap_scores[5],
            'ap_prefer': ap_scores[6],
            'ap_hadamard': ap_scores[7],
            'ap_average': ap_scores[8],
            'ap_L1': ap_scores[9],
            'ap_L2': ap_scores[10],
        }

    @staticmethod
    def _get_rocs(predictors, tp, whole, work_dir, suffix):
        pkl_file = os.path.join(work_dir, 'aucfp' + suffix + ".test")
        if os.path.exists(pkl_file):
            print("loading test_fp pkl_file...")
            with open(pkl_file, 'rb') as fin:
                fp = pickle.load(fin)
        else:
            fp = [LinkPredEval._sample_false(whole) for _ in range(len(tp))]
            fp = set(fp)
            with open(pkl_file, 'wb') as fout:
                pickle.dump(fp, fout)
        auc_scores = []
        ap_scores = []
        for pred in predictors:
            y_true = [True] * len(tp) + [False] * len(fp)
            y_score = [pred.get_sim(*e) for e in list(tp) + list(fp)]
            auc_score = sklearn.metrics.roc_auc_score(y_true, y_score)
            auc_scores.append(auc_score)
            ap_score = average_precision_score(y_true,y_score)
            ap_scores.append(ap_score)
        return auc_scores,ap_scores

    @staticmethod
    def _sample_false(whole):
        n = whole.number_of_nodes()
        while True:
            #u = np.random.choice(n)
            #v = np.random.choice(n)
            u = np.random.randint(0,n-1)
            v = np.random.randint(u+1,n)
            if not whole.has_edge(u, v):
                break
        return u, v


class AlgoBase(object):
    def get_vecs(self, work_dir, train_net, num_nodes, prefix,
                 del_vecfile=False):
        edge_list = os.path.join(work_dir, prefix + '.elist')
        nx.write_edgelist(train_net, edge_list, data=False)
        feats_path = self._learn_vecs(edge_list)
        os.remove(edge_list)
        feats = AlgoBase._read_vecs(feats_path, num_nodes)
        if del_vecfile:
            os.remove(feats_path)
        return feats

    @staticmethod
    def _read_vecs(vec_path, num_nodes):
        with open(vec_path, 'r') as fin:
            num_lines, d = fin.readline().split()
            num_lines, d = int(num_lines), int(d)
            vecs = np.zeros((num_nodes, d), dtype=np.float64)
            for _ in range(num_lines):
                s = fin.readline().split()
                vecs[int(s[0])] = np.asarray([float(x) for x in s[1:]])
        return vecs

    def _learn_vecs(self, _):
        return ''


class AlgoN2VC(AlgoBase):
    def __init__(self, install_dir='./ext'):
        AlgoBase.__init__(self)
        self.main_exe = os.path.join(install_dir,
                                     'snap/examples/node2vec/node2vec')

    def _learn_vecs(self, edge_list_path):
        vec_path = edge_list_path + '.n2v'
        if not os.path.exists(vec_path):
            subprocess.check_call([
                self.main_exe, '-d:128',
                '-p:0.25', '-q:0.25',
                '-i:%s' % edge_list_path, '-o:%s' % vec_path
            ])
        return vec_path


class AlgoN2VPY(AlgoBase):
    def _learn_vecs(self, edge_list_path):
        vec_path = edge_list_path + '.n2v'
        if not os.path.exists(vec_path):
            subprocess.check_call([
                'python2', './ext/node2vec/src/main.py', '--dimensions',
                '128', '--num-walks', '10', '--workers', '1',
                '--p', '0.25', '--q', '0.25',
                '--input', edge_list_path, '--output', vec_path])
        return vec_path


class AlgoDW(AlgoBase):
    def _learn_vecs(self, edge_list_path):
        vec_path = edge_list_path + '.dw'
        if not os.path.exists(vec_path):
            subprocess.check_call([
                'python2', './ext/node2vec/src/main.py', '--dimensions',
                '128', '--num-walks', '10', '--workers', '1',
                '--p', '1.00', '--q', '1.00',
                '--input', edge_list_path, '--output', vec_path])
        return vec_path


class AlgoWK(AlgoBase):
    def _learn_vecs(self, edge_list_path):
        """
        NOTE: Need to modify its source code, due to the "w = 1" problem.
        """
        vec_path = edge_list_path + '.wk'
        if not os.path.exists(vec_path):
            subprocess.check_call([
                './ext/proNet-core/cli/walklets', '-dimensions',
                '128', '-workers', '1', '-walk_times', '80',
                '-train', edge_list_path, '-save', vec_path])
        return vec_path


class AlgoLINE(AlgoBase):
    def __init__(self, install_dir='./ext'):
        AlgoBase.__init__(self)
        line_path = os.path.join(install_dir, "LINE")
        if not os.path.isdir(line_path):
            subprocess.check_call([
                "git", "clone", "https://github.com/tangjianpku/LINE.git",
                line_path
            ])
        line_path = os.path.join(line_path, "linux")
        bin_path = os.path.join(line_path, "bin")
        if not os.path.isdir(bin_path):
            os.mkdir(bin_path)
        for exe in ["line", "reconstruct", "normalize", "concatenate"]:
            subprocess.check_call([
                "g++", "-march=native", "-funroll-loops", "-ffast-math",
                "-Wno-unused-result", "-Ofast",
                os.path.join(line_path, exe + ".cpp"), "-o",
                os.path.join(bin_path, exe), "-pthread", "-lgsl",
                "-lgslcblas"
            ])
        self.bin_path = bin_path

    @staticmethod
    def _line_count(edge_list_path):
        cnt = 0
        with open(edge_list_path) as fin:
            for _ in fin:
                cnt += 1
        return cnt

    def _learn_vecs(self, edge_list_path):
        """
        NOTE: Need to modify its source code, due to the "w = 1" problem.
        """
        n_edges = AlgoLINE._line_count(edge_list_path)
        stage1_path_fmt = edge_list_path + ".lineL%d"
        stage2_path_fmt = edge_list_path + ".lineN%d"
        vec_path = edge_list_path + ".lineV"
        if not os.path.isfile(vec_path):
            for i in (1, 2):
                subprocess.check_call([
                    os.path.join(self.bin_path, "line"), "-train",
                    edge_list_path, "-output", stage1_path_fmt % i,
                    "-binary", "1", "-size", "64", "-order", str(i),
                    "-negative", "5", "-threads", "1",
                    "-samples", str(n_edges * 10 // 10000)
                ])
                subprocess.check_call([
                    os.path.join(self.bin_path, "normalize"), "-input",
                    stage1_path_fmt % i, "-output", stage2_path_fmt % i,
                    "-binary", "1"
                ])
            subprocess.check_call([
                os.path.join(self.bin_path, "concatenate"), "-input1",
                stage2_path_fmt % 1, "-input2", stage2_path_fmt % 2,
                "-output", vec_path, "-binary", "0"
            ])
            for i in (1, 2):
                os.remove(stage1_path_fmt % (i,))
                os.remove(stage2_path_fmt % (i,))
        return vec_path


class AlgoNCRP(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)
        subprocess.check_call(['./compile.sh'])
        self.main_exe = './ncrp'

    def _learn_vecs(self, edge_list_path):
        vec_path = edge_list_path + '.ncrp'
        if not os.path.exists(vec_path):
            subprocess.check_call([
                self.main_exe, edge_list_path, vec_path
            ])
        return vec_path


def main(work_dir, dataset, algocls, prefix, hide_ratio=0.0):
    whole = dataset.network
    label = dataset.label
    if hide_ratio < 1e-6:
        trunc = whole
    else:
        pkl_file = os.path.join(work_dir, prefix + '.trunc_%.2f' % hide_ratio)
        if os.path.exists(pkl_file):
            with open(pkl_file, 'rb') as fin:
                trunc = pickle.load(fin)
        else:
            trunc = LinkPredEval.hide_links(whole, hide_ratio)
            with open(pkl_file, 'wb') as fout:
                pickle.dump(trunc, fout)
    n = whole.number_of_nodes()
    feats = algocls().get_vecs(work_dir, trunc, n, prefix,
                               hide_ratio >= 1e-6)
    if hide_ratio >= 1e-6:
        res = {}
        assert trunc.number_of_nodes() == whole.number_of_nodes()
        assert trunc.number_of_edges() < whole.number_of_edges()
        res.update(
            LinkPredEval.eval(feats, trunc, whole, work_dir,
                              '_%.2f' % hide_ratio))
        print('Link Prediction - %.2f' % hide_ratio)
        for k in sorted(res.keys()):
            print(k, res[k])
        return
    for ratio in range(1, 10):
        ratio = ratio / 10.0
        pkl_file = os.path.join(work_dir, prefix + '.intest_%.2f' % ratio)
        if os.path.exists(pkl_file):
            with open(pkl_file, 'rb') as fin:
                in_test = pickle.load(fin)
        else:
            in_test = np.zeros(n, np.bool)
            n_test = int(n * (1.0 - ratio))
            in_test[np.random.choice(n, n_test, replace=False)] = True
            with open(pkl_file, 'wb') as fout:
                pickle.dump(in_test, fout)
        print(n, ratio, n * ratio)
        res = {}
        res.update(MultiLabelEval.eval(feats[~in_test], label[~in_test],
                                       feats[in_test], label[in_test]))
        print('Classification - %.2f' % ratio)
        for k in sorted(res.keys()):
            print(k, res[k])


if __name__ == '__main__':
    np.random.seed(0)
    wrk_dir = './'

    # Link Prediction
    for hratio in [0.50, 0.40, 0.30, 0.20, 0.10]:
        main(wrk_dir, NetData(wrk_dir, 'cora'), AlgoNCRP, 'tmp', hratio)

    # Node Classification
    main(wrk_dir, NetData(wrk_dir, 'cora'), AlgoNCRP, 'tmp')
