import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import networkx as nx
import os
import operator
from gensim.models import Word2Vec, KeyedVectors
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
import pickle as pkl
import operator

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
    #for se in list(nx.selfloop_edges(G)):
    #    G.remove_edge(se[0],se[1])
    return nx.to_undirected(G)

def load_adjacency_matrix(file, variable_name="network"):
    data = sio.loadmat(file)
    return data[variable_name]

def load_network_matrix(filename,variable_name="network"):
    ext=os.path.splitext(filename)[1]
    if ext == ".mat":
        data = sio.loadmat(filename)
        return data[variable_name]
    else:
        G = nx.read_edgelist(filename, nodetype=int)
        G = G.to_undirected()
        node_number = G.number_of_nodes()
        A = sp.lil_matrix((node_number, node_number))
        for e in G.edges():
            if e[0] != e[1]:
                A[e[0], e[1]] = 1
                A[e[1], e[0]] = 1
        A = sp.csr_matrix(A)
        return A

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
    print("dot product")
    if batch_size is None:
        #matrix_sim = emb.dot(emb.T)
        #return matrix_sim[edges[:,0],edges[:,1]]
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

def cos_similarity(emb,edges):
    emb_norm = normalize(emb,norm='l2',axis=1)
    matrix_sim = emb_norm.dot(emb_norm.T)
    return matrix_sim[edges[:,0],edges[:,1]]

def batch_cos_similarity(emb, edges, batch_size=None):
    print("cosine similarty")
    if batch_size is None:
        return cos_similarity(emb,edges)
    batch_size = int(batch_size)
    n = int(edges.shape[0] // batch_size) # floor and /
    res = []
    emb_norm = normalize(emb,norm='l2',axis=1)
    for i in range(n):
        r = dot_product(emb_norm[edges[i*batch_size:(i+1)*batch_size, 0]], emb_norm[edges[i*batch_size:(i+1)*batch_size, 1]])
        res.append(r)
    a = edges.shape[0]-n*batch_size
    if a > 0:
        res.append(dot_product(emb_norm[edges[n*batch_size:, 0]], emb_norm[edges[n*batch_size:, 1]]))
    return np.hstack(res)

def euclidean_distance(X, Y):
    return np.linalg.norm(X-Y, axis=1)

def batch_euclidean_distance(emb, edges, batch_size=None):
    print("euclidean_distance")
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

def get_edge_embeddings(emb_matrix,edge_list,sim_method):
    embs = []
    for edge in edge_list:
        node1 = edge[0]
        node2 = edge[1]
        emb1 = emb_matrix[node1]
        emb2 = emb_matrix[node2]
        if sim_method == 'avg':
            edge_emb = (emb1+emb2)*0.5 #average
        elif sim_method == 'had':
            edge_emb = np.multiply(emb1, emb2) #hadamard product
        elif sim_method == 'l1':
            edge_emb = np.abs(emb1-emb2) #l1
        elif sim_method == 'l2':
            edge_emb = np.abs(emb1-emb2) ** 2 #l2
        embs.append(edge_emb)
    embs = np.array(embs)
    return embs

def test_dot(emb,e):
    u = e[0]
    v = e[1]
    return np.dot(emb[u],emb[v])

def get_similarity(emb,edges,labels,sim_method,train_lr_edges=None,train_lr_labels=None,batch_size=None):
    if sim_method == 'dp':
        sim = batch_dot_product(emb,edges,batch_size)
        #sim = [test_dot(emb,e) for e in list(edges)]
    elif sim_method == 'cos':
        sim = batch_cos_similarity(emb,edges,1e6)
    elif sim_method == 'euc':
        sim = -batch_euclidean_distance(emb,edges,1e6)
    else:
        test_edge_embs = get_edge_embeddings(emb,edges,sim_method)
        train_edge_embs = get_edge_embeddings(emb,train_lr_edges,sim_method)
        edge_classifier = LogisticRegression(random_state=0)
        edge_classifier.fit(train_edge_embs, train_lr_labels)
        #edge_classifier.fit(test_edge_embs, labels)
        #pro = edge_classifier.predict_proba(test_edge_embs)
        sim = edge_classifier.predict_proba(test_edge_embs)[:, 1]
        #sim = edge_classifier.predict(test_edge_embs)
    return sim


def _plain_bfs(G, source):
    """A fast BFS node generator"""
    seen = set()
    nextlevel = {source}
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if v not in seen:
                yield v
                seen.add(v)
                nextlevel.update(G[v])


def gen_train_test_data(dataset_name, ratio=0.7):
    filename = os.path.join('../datasets', dataset_name, '{}.edgelist'.format(dataset_name))
    graph = load_edgelist(filename)
    get_graph_info(graph)
    graph_train = nx.Graph(graph)
    graph_test = nx.Graph()
    edges = np.random.permutation(list(graph.edges()))
    n_edges = graph.number_of_edges()
    orig_num_cc = nx.number_connected_components(graph)
    print("original number of connected components:{}".format(orig_num_cc))
    num_test_edges = int((1-ratio)*n_edges)
    cnt = 0
    for a, b in edges:
        graph_train.remove_edge(a, b)
        
        reach_from_a = _plain_bfs(graph_train,a)
        if b not in reach_from_a:
            graph_train.add_edge(a, b)
        else:
            graph_test.add_edge(a, b)
            cnt = cnt + 1 
        if cnt == num_test_edges:
            break
        '''
        if nx.number_connected_components(graph_train) > orig_num_cc:
            graph_train.add_edge(a, b)
            continue
        if cnt < num_test_edges:
            cnt = cnt + 1
            graph_test.add_edge(a,b)
        if cnt == num_test_edges:
            break
        '''
    if graph_test.number_of_edges() < num_test_edges:
        print("WARNING: not enough removable edges to perform full train-test split!")
        print("Num. (test) edges requested: ({})".format(num_test_edges))
        print("Num. (test) edges returned: ({})".format(graph_test.number_of_edges()))
    get_graph_info(graph_train)
    get_graph_info(graph_test)
    data_path = os.path.join('../datasets',dataset_name,'{}_{}'.format(dataset_name, ratio))
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    nx.write_edgelist(graph_train, os.path.join(data_path, '{}_{}_train_ncc.edgelist'.format(dataset_name, ratio)), data=False)
    nx.write_edgelist(graph_test, os.path.join(data_path, '{}_{}_test_ncc.edgelist'.format(dataset_name, ratio)), data=False)

def random_gen_train_test_data(dataset_name, ratio=0.7):
    filename = os.path.join('../datasets', dataset_name, '{}.edgelist'.format(dataset_name))
    graph = load_edgelist(filename)
    graph_train = nx.Graph(graph)
    graph_test = nx.Graph()
    num_test_edges = int((1-ratio)*graph.number_of_edges())
    edges = np.random.permutation(list(graph.edges()))
    cnt = 0
    for a, b in edges:
        if cnt < num_test_edges:
            cnt = cnt + 1
            graph_train.remove_edge(a,b)
            graph_test.add_edge(a,b)
        else:
            break
    get_graph_info(graph)
    get_graph_info(graph_train)
    get_graph_info(graph_test)
    assert graph.number_of_nodes() == graph_train.number_of_nodes() , "graph_train do not have the same dimension with graph"
    data_path = os.path.join('../datasets',dataset_name,'{}_{}'.format(dataset_name, ratio))
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    nx.write_edgelist(graph_train, os.path.join(data_path, '{}_{}_train.edgelist'.format(dataset_name, ratio)), data=False)
    nx.write_edgelist(graph_test, os.path.join(data_path, '{}_{}_test.edgelist'.format(dataset_name, ratio)), data=False)


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
    #edges = list(graph.edges())
    #np.random.shuffle(edges)
    nodes = set()
    for (a, b) in edges:
        if a not in nodes or b not in nodes or a == b:
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
    
    assert graph.number_of_nodes() == graph_train.number_of_nodes() , "graph_train do not have the same dimension with graph"
    get_graph_info(graph)
    get_graph_info(graph_train)
    get_graph_info(graph_test)
    data_path = os.path.join('../datasets',dataset_name,'{}_{}'.format(dataset_name, ratio))
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    nx.write_edgelist(graph_train, os.path.join(data_path, '{}_{}_train.edgelist'.format(dataset_name, ratio)), data=False)
    nx.write_edgelist(graph_test, os.path.join(data_path, '{}_{}_test.edgelist'.format(dataset_name, ratio)), data=False)

def hide_links(dataset_name, ratio=0.50):
    filename = os.path.join('../datasets2', dataset_name, '{}.edgelist'.format(dataset_name))
    hide_ratio = 1-ratio
    whole = load_edgelist(filename)
    trunc = nx.Graph()
    trunc = whole.copy()
    for u in range(whole.number_of_nodes()):
        assert trunc.degree(u) > 0
    edges = list(whole.edges())
    np.random.shuffle(edges)
    nums = int(whole.number_of_edges() * hide_ratio)
    cnt = 0
    for (u, v) in edges:
        if u != v and trunc.degree(u) > 1 and trunc.degree(v) > 1:
            if cnt < nums:
                cnt = cnt + 1
            #if np.random.rand() < hide_ratio:
                trunc.remove_edge(u, v)
    for u in range(whole.number_of_nodes()):
        assert trunc.degree(u) > 0
    assert trunc.number_of_nodes() == whole.number_of_nodes()
    print("%d/%d edges kept in TRUNC" % (
        trunc.number_of_edges(), whole.number_of_edges()))
    get_graph_info(whole)
    get_graph_info(trunc)

    whole_edges = set(whole.edges())
    trunc_edges = set(trunc.edges())
    tp = whole_edges - trunc_edges
    g_test = nx.from_edgelist(list(tp))
    get_graph_info(g_test)

    data_path = os.path.join('../datasets2',dataset_name,'{}_{}'.format(dataset_name, ratio))
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    nx.write_edgelist(trunc, os.path.join(data_path, '{}_{}_train.edgelist'.format(dataset_name, ratio)), data=False)








