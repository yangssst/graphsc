import json
import os
import math
import warnings
import torch
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from Graph_Construction import Graph
from gcn import GCN, get_embedding, get_loss, fit
from utils import (
    get_sample_implied_labels,
    check_reconstruction,
    get_sample_expressions,
    PCA_decomposition,
    Eigengene_significance,
)

warnings.filterwarnings("ignore")
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def compute_clustering_metrics(valid_embedding, valid_labels):
    metrics = {'ch_index': -1, 'sc_index': -1, 'pearson_sc_index': -1, 'cosine_sc_index': -1}
    if len(np.unique(valid_labels)) <= 1:
        print("Cannot compute clustering metrics due to only one cluster.")
        return metrics
    metrics['ch_index'] = calinski_harabasz_score(valid_embedding, valid_labels)
    metrics['sc_index'] = silhouette_score(valid_embedding, valid_labels)
    try:
        pearson_distance_matrix = 1 - np.corrcoef(valid_embedding)
        metrics['pearson_sc_index'] = silhouette_score(pearson_distance_matrix, valid_labels, metric='precomputed')
    except ValueError:
        pass
    print(f"CH = {metrics['ch_index']:.2f}, SC = {metrics['sc_index']:.2f}, "
          f"Pearson_SC = {metrics['pearson_sc_index']:.2f}")
    return metrics

def module_detection(embedding, output_folder, graph_data, embname, k_max=None):
    if k_max is None:
        k_max = int(math.sqrt(len(embedding))) * 2
    spectral = SpectralClustering(n_clusters=k_max, affinity='nearest_neighbors', assign_labels='kmeans').fit(embedding)
    labels = spectral.labels_
    valid_idx = np.isin(labels, np.unique(labels)[np.bincount(labels) >= 4])
    valid_embedding = embedding[valid_idx]
    valid_labels = labels[valid_idx]

    module_es, gene_cluster_mapping = {}, []
    for g in np.unique(valid_labels):
        indices = np.where(labels == g)[0]
        cluster_gene_names = [graph_data.gene_names[i] for i in indices]
        expressions = get_sample_expressions(graph_data.gene_names, cluster_gene_names,
                                             graph_data.expressions, graph_data.sample_names)
        eigengene = PCA_decomposition(np.transpose(expressions))
        module_es[g] = Eigengene_significance(eigengene, get_sample_implied_labels(graph_data.sample_names))

    return module_es

def main(data, struct, check_recon, epochs_limit=100, batch_size=32, device='cpu'):

    struct[0] = data['train'].N
    model = GCN(struct).to(device)

    train_data = data['train']
    test_data = data['test']
    embedding = None
    while (True):
        mini_batch = train_data.sample(batch_size, do_shuffle=False)
        if embedding is None:
            embedding = get_embedding(model, mini_batch, device)
        else:
            embedding = np.vstack((embedding, get_embedding(model, mini_batch, device)))
        if train_data.epoch_end:
            break

    if check_recon:
        print("Epoch 0 reconstruction:", check_reconstruction(embedding, train_data, check_recon))

    best_mes = -np.inf
    for epoch in range(epochs_limit):
        loss, tr_emb = 0, None
        while(True):
            batch = train_data.sample(batch_size, do_shuffle=False)
            loss += get_loss(model, batch, device)
            if tr_emb is None:
                tr_emb = get_embedding(model, batch, device)
            else:
                tr_emb = np.vstack((tr_emb, get_embedding(model, batch, device)))
            if train_data.epoch_end:
                break

        os.makedirs('output', exist_ok=True)
        tr_module_es = module_detection(tr_emb, 'output', train_data, f'tr_emb_{epoch}')
        if max(tr_module_es.values()) > best_mes:
            best_mes = max(tr_module_es.values())
        batch = train_data.sample(batch_size)
        fit(model, batch, mes=best_mes, device=device)
        print(f"Epoch {epoch}: loss = {loss:.4f}  ")

        # test
        te_loss, te_emb = 0, None
        while (True):
            batch = test_data.sample(batch_size, do_shuffle=False)
            te_loss += get_loss(model, batch, device)
            if te_emb is None:
                te_emb = get_embedding(model, batch, device)
            else:
                te_emb = np.vstack((te_emb, get_embedding(model, batch, device)))
            if test_data.epoch_end:
                break
        te_module_es = module_detection(te_emb, 'output', test_data, f'te_emb_{epoch}')
        print(f"Epoch {epoch}: test loss = {te_loss:.4f} ")
        if epoch % 10 == 0:
            if check_recon:
                print(f"Epoch {epoch} test reconstruction:", check_reconstruction(te_emb, test_data, check_recon))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_file = 'data/coad_train.csv'
    test_file = 'data/coad_test.csv'
    struct = [-1, 400, 200, 25]
    check_recon = [10,100,1000,1500]
    
    data = {}
    data['train'] = Graph(train_file, top_k_genes=2000)
    data['test'] = Graph(test_file, top_k_genes=2000)
    
    main(data, struct, check_recon, epochs_limit=200, batch_size=32, device=device)