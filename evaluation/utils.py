import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import networkx as nx
import os
import operator
from gensim.models import Word2Vec, KeyedVectors
import pickle as pkl

def load_w2v_feature(file):
    with open(file, "rb") as f:
        nu = 0
        for line in f:
            content = line.strip().split()
            nu += 1
            if nu == 1:
                n, d = int(content[0]), int(content[1])
                feature = [[] for i in range(n)]
                continue
            index = int(content[0])
            for x in content[1:]:
                feature[index].append(float(x))
            if nu % 10000000 == 0:
                print("read {} line from w2v feature file".format(nu))
    return np.array(feature, dtype=np.float32)

def save_wv(filename, data):
    with open(filename, 'w') as f:
        nums, embedding_size = data.shape
        print(nums, embedding_size, file=f)
        for j in range(nums):
            print(j, *data[j], file=f)

def load_edgelist(filename):
    G = nx.read_edgelist(filename,nodetype=int)
    return nx.to_undirected(G)

def load_adjacency_matrix(file, variable_name="network"):
    data = sio.loadmat(file)
    return data[variable_name]

def load_network_matrix(filename,variable_name="network"):
    ext=os.path.splitext(filename)[1]
    if ext == ".mat":
        data = sio.loadmat(filename)
        return data[variable_name]
    elif ext == ".edgelist":
        G = nx.read_edgelist(filename,nodetype=int)
        G = nx.to_undirected(G)
        return nx.to_scipy_sparse_matrix(G)

def load_label(file, variable_name="group"):
    if file.endswith(".tsv") or file.endswith(".txt"):
        data = np.loadtxt(file).astype(np.int32)
        label = sp.csr_matrix(([1] * data.shape[0], (data[:, 0], data[:, 1])), dtype=np.bool_)
        sp.save_npz("label.npz", label)
        return label
    elif file.endswith(".npz"):
        return sp.load_npz(file)
    else:
        data = sio.loadmat(file)
        label = data[variable_name].tocsr().astype(np.bool_)
        print(label.shape, label.dtype)
        return label
    label = data[variable_name].todense().astype(np.int32)
    label = np.array(label)
    return label

def load_embeddings(file):
    ext = os.path.splitext(file)[1]
    if ext == ".npy":
        embedding = np.load(file)
    elif ext == ".pkl":
        with open(file,"rb") as f:
            embedding = pkl.load(f)
    else:
        embedding = load_w2v_feature(file)
    return embedding    

def get_graph_info(G):
    print("Graph info: num of nodes-----%d,num of edges-----%d"%(G.number_of_nodes(), G.number_of_edges()))

def dot_product(X, Y):
    return np.sum(X*Y, axis=1)

def batch_dot_product(emb, edges, batch_size=None):
    if batch_size is None:
        return dot_product(emb[edges[:, 0]], emb[edges[:, 1]])
    batch_size = int(batch_size)
    n = int(edges.shape[0] // batch_size) # floor and /
    res = []
    for i in range(n):
        r = dot_product(emb[edges[i*batch_size:(i+1)*batch_size, 0]], emb[edges[i*batch_size:(i+1)*batch_size, 1]])
        res.append(r)
    a = edges.shape[0]-n*batch_size
    if a > 0:
        res.append(dot_product(emb[edges[n*batch_size:, 0]], emb[edges[n*batch_size:, 1]]))
    return np.hstack(res)

def euclidean_distance(X, Y):
    print("euclidean_distance")
    return np.linalg.norm(X-Y, axis=1)

def batch_euclidean_distance(emb, edges, batch_size=None):
    print("batch_euclidean_distance")
    if batch_size is None:
        return euclidean_distance(emb[edges[:, 0]], emb[edges[:, 1]])
    batch_size = int(batch_size)
    n = int(edges.shape[0] // batch_size) # floor and /
    res = []
    for i in range(n):
        r = euclidean_distance(emb[edges[i*batch_size:(i+1)*batch_size, 0]], emb[edges[i*batch_size:(i+1)*batch_size, 1]])
        res.append(r)
    a = edges.shape[0]-n*batch_size
    if a > 0:
        res.append(euclidean_distance(emb[edges[n*batch_size:, 0]], emb[edges[n*batch_size:, 1]]))
    return np.hstack(res)


def split_dataset(dataset_name, ratio=0.7):
    filename = os.path.join('../datasets', dataset_name, '{}.edgelist'.format(dataset_name))
    #A = load_adjacency_matrix(filename)
    #print(A.nnz)
    #graph = nx.from_scipy_sparse_matrix(A)
    #path = "../datasets/name/name.edgelist"
    #nx.write_edgelist(graph,path,data=False)
    graph = load_edgelist(filename)
    graph_train = nx.Graph()
    graph_test = nx.Graph()
    edges = np.random.permutation(list(graph.edges()))
    nodes = set()
    for a, b in edges:
        if a not in nodes or b not in nodes:
            graph_train.add_edge(a, b)
            nodes.add(a)
            nodes.add(b)
        else:
            graph_test.add_edge(a, b)
    assert len(nodes) == graph.number_of_nodes()
    assert len(nodes) == graph_train.number_of_nodes()
    num_test_edges = int((1-ratio)*graph.number_of_edges())
    now_number = graph_test.number_of_edges()
    if num_test_edges < now_number:
        test_edges = list(graph_test.edges())
        graph_train.add_edges_from(test_edges[:now_number-num_test_edges])
        graph_test.remove_edges_from(test_edges[:now_number-num_test_edges])

    get_graph_info(graph)
    get_graph_info(graph_train)
    get_graph_info(graph_test)

    data_path = os.path.join('../datasets',dataset_name,'{}_{}'.format(dataset_name, ratio))

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    nx.write_edgelist(graph_train, os.path.join(data_path, '{}_{}_train.edgelist'.format(dataset_name, ratio)), data=False)
    nx.write_edgelist(graph_test, os.path.join(data_path, '{}_{}_test.edgelist'.format(dataset_name, ratio)), data=False)






