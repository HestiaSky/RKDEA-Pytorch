"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import math
import json

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import random
import torch.nn.functional as F
from torchtext.data import Dataset, BucketIterator, Field, Example
from torch.utils.data import TensorDataset, DataLoader


def load_data(args):
    data = load_data_ea(args)

    return data


# ############### FEATURES PROCESSING ############### #


def process(x, adj, norm_x, norm_adj):
    if norm_x:
        x = normalize(x)
    if norm_adj:
        adj = normalize(adj)
    return x, adj


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# ############### Entity Alignment Dataloader ############### #
# calculate relation sets
def rfunc(e, KG):
    head = {}
    tail = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = list([tri[0]])
            tail[tri[1]] = list([tri[2]])
        else:
            cnt[tri[1]] += 1
            head[tri[1]].append(tri[0])
            tail[tri[1]].append(tri[2])
    r_num = len(head)
    head_r = np.zeros((e, r_num))
    tail_r = np.zeros((e, r_num))
    for tri in KG:
        head_r[tri[0]][tri[1]] = 1
        tail_r[tri[2]][tri[1]] = 1

    return head, tail, head_r, tail_r


# get a dense adjacency matrix and degree
def get_matrix(e, KG):
    degree = [1] * e
    for tri in KG:
        if tri[0] != tri[2]:
            degree[tri[0]] += 1
            degree[tri[2]] += 1
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 1
        else:
            pass
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = 1
        else:
            pass
    for i in range(e):
        M[(i, i)] = 1
    return M, degree


# get a sparse tensor based on relational triples
def get_sparse_tensor(e, KG):
    print('getting a sparse tensor...')
    M, degree = get_matrix(e, KG)
    row = []
    col = []
    val = []
    for fir, sec in M:
        row.append(fir)
        col.append(sec)
        val.append(M[(fir, sec)] / math.sqrt(degree[fir]) / math.sqrt(degree[sec]))
    M = sp.coo_matrix((val, (row, col)), shape=(e, e))
    return M


def get_features(lang):
    print('adding the primal input layer...')
    with open(file='data/dbp15k/' + lang + '_en/' + lang + '_vectorList.json', mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
        print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
    ent_embeddings = torch.Tensor(embedding_list)
    return sp.coo_matrix(F.normalize(ent_embeddings, 2, 1))


# load a file and return a list of tuple containing $num integers in each line
def loadfile(fn, num=1):
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def load_data_ea(args):
    lang = args.dataset  # zh_en | ja_en | fr_en
    e1 = 'data/dbp15k/' + lang + '/ent_ids_1'
    e2 = 'data/dbp15k/' + lang + '/ent_ids_2'
    r1 = 'data/dbp15k/' + lang + '/rel_ids_1'
    r2 = 'data/dbp15k/' + lang + '/rel_ids_2'
    ill = 'data/dbp15k/' + lang + '/ref_ent_ids'
    ill_r = 'data/dbp15k/' + lang + '/ref_r_ids'
    kg1 = 'data/dbp15k/' + lang + '/triples_1'
    kg2 = 'data/dbp15k/' + lang + '/triples_2'

    e = len(set(loadfile(e1, 1)) | set(loadfile(e2, 1)))
    r = len(set(loadfile(r1, 1)) | set(loadfile(r2, 1)))
    ILL = loadfile(ill, 2)
    illL = len(ILL)
    np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * 3])
    test = np.array(ILL[illL // 10 * 3:])
    test_r = loadfile(ill_r, 2)
    KG = loadfile(kg1, 3) + loadfile(kg2, 3)

    features = get_features(lang[0:2])
    features = sparse_mx_to_torch_sparse_tensor(features)
    M = get_sparse_tensor(e, KG)
    M = sparse_mx_to_torch_sparse_tensor(M)
    head, tail, head_r, tail_r = rfunc(e, KG)
    feat = features.to_dense()
    features_r = torch.FloatTensor(r, len(feat[0]))
    for rel in range(r):
        features_r[rel] = (torch.sum(feat[tail[rel]], 0) - torch.sum(feat[head[rel]], 0)) / len(head[rel])
    features_r = features_r.to_sparse()
    data = {'x': features, 'adj': M, 'r': features_r, 'train': train, 'test': test, 'test_r': test_r, 'triple': KG,
            'head': head, 'tail': tail, 'head_r': head_r, 'tail_r': tail_r,
            'idx_x': torch.LongTensor(range(features.shape[0])), 'idx_r': torch.LongTensor(range(features_r.shape[0]))}
    if args.model == 'RKDEA':
        print('loading TransE embeddings...')
        data['emb'] = torch.from_numpy(np.load(f'data/dbp15k/{args.dataset}/TransE_embeddings.npy'))
        print(data['emb'].shape[0], 'rows', data['emb'].shape[1], 'columns')
    return data


