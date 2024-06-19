import numpy as np
import skfuzzy as fuzz
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score, jaccard_score, adjusted_mutual_info_score,mutual_info_score
from scipy.special import comb
def hubert_index(labels_true, labels_pred):
    n = len(labels_true)
    sum_agree = sum_disagree = 0

    for i in range(n):
        for j in range(i+1, n):
            agree = (labels_true[i] == labels_true[j]) == (labels_pred[i] == labels_pred[j])
            disagree = (labels_true[i] == labels_true[j]) != (labels_pred[i] == labels_pred[j])
            sum_agree += agree
            sum_disagree += disagree

    gamma_statistic = (sum_agree - sum_disagree) / comb(n, 2)
    return gamma_statistic


def jaccard_index(labels_true, labels_pred):
    # Create a binary matrix for each label set
    n = len(labels_true)
    a = np.zeros((n, n))
    b = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            a[i, j] = labels_true[i] == labels_true[j]
            b[i, j] = labels_pred[i] == labels_pred[j]

    # Flatten the matrices and calculate Jaccard Index
    return jaccard_score(a.flatten(), b.flatten())


def compute_flicm(data, n_clusters, m=2, max_iter=100, error=1e-5):
    """
    Fuzzy Local Information C-Means (FLICM) clustering algorithm.

    Parameters:
    - data: Input data, shape (n_samples, n_features)
    - n_clusters: Number of clusters
    - m: Fuzziness parameter (default=2)
    - max_iter: Maximum number of iterations (default=100)
    - error: Stopping criterion (default=1e-5)

    Returns:
    - cntr: Final cluster centers
    - u: Final fuzzy partition matrix
    """
    # Initialize variables
    n_samples = data.shape[0]
    u = np.random.rand(n_clusters, n_samples)
    u = u / np.sum(u, axis=0, keepdims=True)

    for iteration in range(max_iter):
        u_old = u.copy()

        # Compute cluster centers
        um = u ** m
        cntr = np.dot(um, data) / np.sum(um, axis=1, keepdims=True)

        # Compute distances and local spatial information
        dist = np.zeros((n_clusters, n_samples))
        for k in range(n_clusters):
            dist[k] = np.linalg.norm(data - cntr[k], axis=1)

        # Update fuzzy partition matrix
        for k in range(n_clusters):
            u[k] = 1.0 / np.sum((dist[k] / dist) ** (2 / (m - 1)), axis=0)

        # Check for convergence
        if np.linalg.norm(u - u_old) < error:
            break

    return cntr, u
def main(data,Label,k,NCEI,ARI,FM,JI,MI,HI):
    # Perform Fuzzy C-Means clustering
    # Perform FLICM clustering
    cntr, u = compute_flicm(data, k)

    # Extract cluster membership values
    predicted_labels = np.argmax(u, axis=0)
    ari = adjusted_rand_score(Label.flatten(), predicted_labels)
    fm = fowlkes_mallows_score(Label.flatten(), predicted_labels)
    ji = jaccard_index(Label.flatten(), predicted_labels)
    hubert = hubert_index(Label.flatten(), predicted_labels)
    mi = mutual_info_score(Label.flatten(), predicted_labels)
    threshold = 0
    outliers = predicted_labels[np.abs(Label.flatten()) > threshold]
    alpha = [1 - sum(outliers) / len(data)]
    beta = 1 - np.array(alpha)
    nn = np.array(alpha) * ari +np.array(beta) *ji #NCEI
    NCEI.append(nn)
    MI.append(mi)
    ARI.append(ari)
    FM.append(fm)
    JI.append(ji)

    HI.append(hubert)
