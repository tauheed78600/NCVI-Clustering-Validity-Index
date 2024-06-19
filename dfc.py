import numpy as np
import skfuzzy as fuzz
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score, jaccard_score, adjusted_mutual_info_score,mutual_info_score
from scipy.special import comb
from scipy.linalg import norm
from scipy.spatial.distance import cdist

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
def reyi_entropy(labels):
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    counts = np.bincount(labels)
    probs = counts[np.nonzero(counts)] / n_labels
    n_classes = len(probs)
    if n_classes <= 1:
        return 0
    return - np.sum(probs * np.log(probs)) / np.log(n_classes)
class DFC_m:
    def __init__(self, n_clusters=10, max_iter=150, m=2, error=1e-5, random_state=42):
        self.u, self.centers = None, None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error
        self.random_state = random_state

    def fit(self, X):
        N = X.shape[0]
        C = self.n_clusters
        centers = []

        # u = np.random.dirichlet(np.ones(C), size=N)
        r = np.random.RandomState(self.random_state)
        u = r.rand(N,C)

        u = u / np.tile(u.sum(axis=1)[np.newaxis].T,C)

        iteration = 0
        while iteration < self.max_iter:
            u2 = u.copy()

            centers = self.next_centers(X, u)
            u = self.next_u(X, centers)
            iteration += 1

            # Stopping rule
            if norm(u - u2) < self.error:
                break

        self.u = u
        self.centers = centers
        return self

    def next_centers(self, X, u):
        um = u ** self.m
        return (X.T @ um / np.sum(um, axis=0)).T

    def next_u(self, X, centers):
        return self._predict(X, centers)

    def _predict(self, X, centers):
        power = float(2 / (self.m - 1))
        temp = cdist(X, centers) ** power
        # alpha=reyi_entropy(temp)
        denominator_ = temp.reshape((X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
        denominator_ = temp[:, :, np.newaxis] / denominator_

        return 1 / denominator_.sum(2)

    def predict(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        u = self._predict(X, self.centers)
        return np.argmax(u, axis=-1)

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


def main(data,Label,k,NCEI,ARI,FM,JI,MI,HI):
    data = np.array(data)
    # Perform Fuzzy C-Means clustering
    data = StandardScaler().fit_transform(data)

    # Step 2: Define a simple neural network for feature extraction
    class FeatureExtractor(tf.keras.models.Model):
        def __init__(self):
            super(FeatureExtractor, self).__init__()
            self.dense1 = tf.keras.layers.Dense(64, activation='relu')
            self.dense2 = tf.keras.layers.Dense(32, activation='relu')
            self.dense3 = tf.keras.layers.Dense(2, activation=None)  # Reduced dimensionality to 2 for visualization

        def call(self, x):
            x = self.dense1(x)
            x = self.dense2(x)
            return self.dense3(x)

    # Step 3: Train the neural network
    model = FeatureExtractor()
    model.compile(optimizer='adam', loss='mse')

    # Dummy target, not used in unsupervised learning
    dummy_target = np.zeros((data.shape[0], 2))

    model.fit(data, dummy_target, epochs=10, batch_size=32)

    # Step 4: Extract features using the trained model
    features = model.predict(data)

    # Step 5: Apply fuzzy c-means clustering on the extracted features
    n_clusters = 4
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None
    )

    # Extract cluster membership values
    # cluster_membership = np.argmax(u, axis=0)

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
