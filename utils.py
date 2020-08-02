# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import scipy.io
import scipy.sparse as sparse
from scipy.sparse import csgraph

def direct_compute_deepwalk_matrix(A):
    window = 1
    b = 1.0
    n = A.shape[0]
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    # X = D^{-1/2} A D^{-1/2}
    X = sparse.identity(n) - L
    S = np.zeros_like(X)
    X_power = sparse.identity(n)
    for i in range(window):
        print("Compute matrix %d-th power"%(i+1))
        X_power = X_power.dot(X)
        S += X_power
    S *= vol / window / b
    D_rt_inv = sparse.diags(d_rt ** -1)
    M = D_rt_inv.dot(D_rt_inv.dot(S).T)
    Y = np.log(np.where(M>1,M,1))
    return Y


def process_adj(A,order):
    M = np.zeros_like(A)
    for t in range(order):
        M = M + A**(t+1)
    M = (1/order)*M
    return M

def read_edgelist(cfg):
    print('read edgelist...')
    with open(cfg.path,'r') as f:
        views = []
        adj_mat = []
        for line in f:
            ls = line.strip().split()
            edgelist = ls[0]
            directed = bool(ls[1])
            weighted = bool(ls[2])
            if weighted:
                G = nx.read_edgelist(edgelist, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
            else:
                G = nx.read_edgelist(edgelist, nodetype=int, create_using=nx.DiGraph())
                for edge in G.edges():
                    G[edge[0]][edge[1]]['weight'] = 1
            if not directed:
                G = G.to_undirected()
            views.append(G)
            A = nx.to_numpy_matrix(G,cfg.node_list)
            if cfg.self_loop:
                A = A + np.identity(A.shape[0])
            if cfg.row_norm:
                A = A/A.sum(1)
            if cfg.dw_matrix == 'yang':
                A = np.where(A>1,1,A)
                A = process_adj(A,cfg.order)
            else:
                A = direct_compute_deepwalk_matrix(A)
                A = A/np.max(A)
            adj_mat.append(A)
    f.close()
    return views,adj_mat

def read_node(cfg):
    print('read nodes...')
    with open(cfg.nodes,'r') as f:
        node_list = []
        look_up   = dict()
        k = 0
        for line in f:
            ls = line.strip().split()
            node_list.append(int(ls[0]))
            look_up[int(ls[0])] = k
            k += 1
    f.close()
    print('Number of nodes = ',len(node_list))
    return node_list,look_up

def read_label(cfg):
    node_label = {}
    if cfg.multilabel==False:
        with open(cfg.labels,'r') as f:
            for line in f:
                ls = line.strip().split()
                node = int(ls[0])
                label = int(ls[1])
                node_label[node] = label
        f.close()
    else:
        with open(cfg.labels,'r') as f:
            for line in f:
                ls = line.strip().split()
                node = int(ls[0])
                label = [int(x) for x in ls[1:]]
                node_label[node] = label
    return node_label