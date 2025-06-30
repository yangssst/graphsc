import numpy as np
import pandas as pd
import random
from scipy.sparse import dok_matrix


def prepare_data(file_path, top_k_genes=2000):
    df = pd.read_csv(file_path, index_col=0)
    gene_stds = df.std(axis=1)
    top_genes = gene_stds.nlargest(top_k_genes).index
    top_expr = df.loc[top_genes].values
    return df.columns.tolist(), top_genes.tolist(), top_expr


def create_edge_index(expr_matrix, st=7, ht=0.01):
    corr = np.corrcoef(expr_matrix)
    edges = []
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            strength = abs(corr[i, j]) ** st
            if strength >= ht:
                edges.append((i, j))
    return np.array(edges, dtype=int)


class Graph:
    def __init__(self, file_path, top_k_genes=2000, ng_sample_ratio=0.0, st=7, ht=0.01):
        self.sample_names, self.gene_names, self.expressions = prepare_data(file_path, top_k_genes)
        self.N = len(self.gene_names)

        edge_index = create_edge_index(self.expressions, st=st, ht=ht)
        self.E = len(edge_index)
        self.adj_matrix = dok_matrix((self.N, self.N), dtype=int)

        for i, j in edge_index:
            self.adj_matrix[i, j] = 1
            self.adj_matrix[j, i] = 1

        self.adj_matrix = self.adj_matrix.tocsr()

        if ng_sample_ratio > 0:
            self._add_negative_samples(int(ng_sample_ratio * self.E))

        self.order = np.arange(self.N)
        self.ptr = 0
        self.epoch_end = False

        print(f"Vertices: {self.N}, Edges: {self.E}, NegSampleRatio: {ng_sample_ratio:.3f}")

    def _add_negative_samples(self, n_samples):
        print("Adding negative samples...")
        count = 0
        while count < n_samples:
            i, j = random.randint(0, self.N - 1), random.randint(0, self.N - 1)
            if i != j and self.adj_matrix[i, j] == 0:
                self.adj_matrix[i, j] = -1
                self.adj_matrix[j, i] = -1
                count += 1
        print("Negative sampling done.")

    def sample(self, batch_size, do_shuffle=True):
        if self.epoch_end:
            self.order = np.random.permutation(self.N) if do_shuffle else np.arange(self.N)
            self.ptr = 0
            self.epoch_end = False

        end = min(self.ptr + batch_size, self.N)
        indices = self.order[self.ptr:end]

        X = self.adj_matrix[indices].toarray()
        adj = X[:, indices]

        self.ptr = end
        if self.ptr >= self.N:
            self.epoch_end = True

        return {"X": X, "adj": adj}
