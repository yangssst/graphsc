import copy
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats

def get_Similarity(result):
    print("getting similarity...")
    return np.dot(result, result.T)

def check_reconstruction(embedding, graph_data, check_index):
    def get_precisionK(embedding, data, max_index):
        print("get precisionK...")
        similarity = get_Similarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        precisionK = []
        sortedInd = sortedInd[::-1]
        for ind in sortedInd:
            x = ind // data.N
            y = ind % data.N
            count += 1
            if (data.adj_matrix[x].toarray()[0][y] == 1 or x == y):
                cur += 1
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
        return precisionK

    precisionK = get_precisionK(embedding, graph_data, np.max(check_index))
    ret = []
    for index in check_index:
        print("precisonK[%d] %.2f" % (index, precisionK[index - 1]))
        ret.append(precisionK[index - 1])
    return ret

def get_sample_expressions(gene_names, target_genes, expressions, sample_names):
    selected = [[] for _ in range(len(sample_names))]
    for i, gene in enumerate(gene_names):
        if gene in target_genes:
            for j in range(len(sample_names)):
                selected[j].append(expressions[i][j])
    return copy.deepcopy(selected)

def get_sample_implied_labels(sample_ids):
    labels = []
    for sid in sample_ids:
        sample_type = int(sid[13:15])
        labels.append(1 if sample_type >= 10 else 0)
    return labels

def PCA_decomposition(expression_matrix):
    pca = PCA(n_components=1)
    pca.fit(expression_matrix)
    return pca.components_[0]

def Eigengene_significance(eigengene, sample_labels):
    corr, _ = stats.pearsonr(eigengene, sample_labels)
    return abs(corr)
