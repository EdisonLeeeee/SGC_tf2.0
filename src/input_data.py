import scipy.sparse as sp
import numpy as np

from sklearn.model_selection import train_test_split
from time import perf_counter


class Dataset(object):
    def __init__(self, dataset, seed=None):
        self.name = dataset
        self.adj, self.features, self.labels = self.load_npz(dataset)
        
        self.n_nodes = self.adj.shape[0]
        self.n_features = self.features.shape[1]
        self.n_classes = self.labels.max() + 1
        
        # Symmetrization and add self-loop
        self.adj += self.adj.T + sp.eye(self.n_nodes)
        self.adj[self.adj>1] = 1.

        self.adj_norm = preprocess_adj(self.adj)
        self.feat, self.precompute_time = sgc_precompute(self.features, self.adj_norm)

        feat_dense = self.feat.todense()

        self.idx_train, self.idx_val, self.idx_test = train_val_test_split_tabular(
            self.n_nodes, stratify=self.labels, random_state=seed)

        self.train_feat = feat_dense[self.idx_train]
        self.val_feat = feat_dense[self.idx_val]
        self.test_feat = feat_dense[self.idx_test]
        self.y_train = self.labels[self.idx_train]
        self.y_val = self.labels[self.idx_val]
        self.y_test = self.labels[self.idx_test]

    def load_npz(self, file_name):
        if not file_name.endswith('.npz'):
            file_name += '.npz'
        with np.load(f'../Data/{file_name}', allow_pickle=True) as loader:
            loader = dict(loader)
            adj_matrix = sp.csr_matrix(
                (loader['adj_data'], loader['adj_indices'],
                 loader['adj_indptr']),
                shape=loader['adj_shape'])

            if 'attr_data' in loader:
                attr_matrix = sp.csr_matrix(
                    (loader['attr_data'], loader['attr_indices'],
                     loader['attr_indptr']),
                    shape=loader['attr_shape'])
            elif 'attr_matrix' in loader:
                '''This is for the dataset with continuous feature.'''
                from sklearn.preprocessing import StandardScaler
                attr_matrix = loader['attr_matrix']
                scaler = StandardScaler()
                scaler.fit(attr_matrix)
                attr_matrix = scaler.transform(attr_matrix)
                attr_matrix = sp.csr_matrix(attr_matrix)

            else:
                attr_matrix = None
            labels = loader.get('labels')
        return adj_matrix, attr_matrix, labels


def train_val_test_split_tabular(N,
                                 train_size=0.1,
                                 val_size=0.1,
                                 test_size=0.8,
                                 stratify=None,
                                 random_state=None):

    idx = np.arange(N)
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size +
                                                               val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
        idx_train, idx_val = train_test_split(idx_train_and_val,
                                              random_state=random_state,
                                              train_size=(train_size / (train_size + val_size)),
                                              test_size=(val_size / (train_size + val_size)),
                                              stratify=stratify)

    return idx_train, idx_val, idx_test


def sgc_precompute(features, adj, degree=2):
    adj = adj.tocoo()
    features = features.tocoo()
    t = perf_counter()
    for _ in range(degree):
        features = adj.dot(features)
    precompute_time = perf_counter() - t

    return features, precompute_time

# def preprocess_features(features):
#     """Row-normalize feature matrix"""
#     rowsum = np.array(features.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     features = r_mat_inv.dot(features)
#     return features


def preprocess_adj(adj):
    """Normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)