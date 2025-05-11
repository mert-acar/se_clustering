import numpy as np
from scipy.sparse.linalg import svds
from scipy.optimize import linear_sum_assignment
from sklearn import cluster
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    homogeneity_score,
    completeness_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)
from typing import Dict


def scores(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    assert y_true.shape == y_pred.shape, "Input arrays must have the same shape"
    return {
        "accuracy": clustering_accuracy(y_true, y_pred),
        "homogeneity": float(homogeneity_score(y_true, y_pred)),
        "completeness": float(completeness_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
    }


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    assert y_true.shape == y_pred.shape, "Input arrays must have the same shape"

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    n_true = len(true_labels)
    n_pred = len(pred_labels)
    cost_matrix = np.zeros((n_true, n_pred))

    for i, true_label in enumerate(true_labels):
        for j, pred_label in enumerate(pred_labels):
            matches = np.sum((y_true == true_label) & (y_pred == pred_label))
            cost_matrix[i, j] = -matches

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    label_mapping = dict(zip(pred_labels[col_ind], true_labels[row_ind]))
    y_pred_aligned = np.array([label_mapping[label] for label in y_pred])
    accuracy = np.sum(y_true == y_pred_aligned) / len(y_true)
    return float(accuracy)


def block_diagonalize(
    coeff: np.ndarray, n_clusters: int, n_dims: int, alpha: int = 8
) -> np.ndarray:
    coeff = 0.5 * (coeff + coeff.T)
    n = coeff.shape[0]
    coeff = coeff - np.diag(np.diag(coeff)) + np.eye(n, n)
    rank = min(n_dims * n_clusters + 1, n - 1)
    u, s, _ = svds(coeff, rank, v0=np.ones(n))
    u = u[:, ::-1]  # type: ignore
    s = np.sqrt(s[::-1])
    s = np.diag(s)
    u = u.dot(s)
    u = normalize(u, norm="l2", axis=1)
    z = u.dot(u.T)  # type: ignore
    z = z * (z > 0)
    l = np.abs(z**alpha)
    l = l / l.max()
    l = (l + l.T) / 2
    return l


def thrC(C, alpha):
    if alpha < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while stop == False:
                csum = csum + S[t, i]
                if csum > alpha * cL1:
                    stop = True
                    Cp[Ind[0 : t + 1, i], i] = C[Ind[0 : t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def spectral_clustering(
    coeff: np.ndarray, num_classes: int, dims: int, alpha: int = 8, ro: float = 0.12
) -> np.ndarray:
    coeff = thrC(coeff, ro)
    block_coeff = block_diagonalize(coeff, num_classes, dims, alpha)
    spectral = cluster.SpectralClustering(
        n_clusters=num_classes,
        eigen_solver="arpack",
        affinity="precomputed",
        assign_labels="discretize",
    )
    return spectral.fit_predict(block_coeff)
