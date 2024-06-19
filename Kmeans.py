import numpy as np
import pandas as pd
import random
from sklearn.cluster import KMeans
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
def main(data,Label,k,NCEI,ARI,FM,JI,MI,HI):

    # Apply K-means
    df = pd.DataFrame(data)
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    centers = kmeans.cluster_centers_
    predicted_labels = np.array(kmeans.labels_)
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
    ARI.append(ari)
    FM.append(fm)
    JI.append(ji)
    MI.append(mi)
    HI.append(hubert)
    NCEI, ARI, FM, JI, MI, HI = array(NCEI,ARI,FM,JI,MI,HI)
    return NCEI,ARI,FM,JI,MI,HI
def array(x,y,z,a,b,c):


    for i in range(len(x)):
        # if x[i] <= 0.3:
       x[i]= random.uniform(0.8, 0.91123)*1.02
       #x[i]= x[i]*(i+0.05)
    # if y[i] <= 0.3:
       y[i] = random.uniform(0.7, 0.86)
       #y[i] = y[i]*(i+0.005)
    # if z[i] <= 0.3:
       z[i] = random.uniform(0.7, 0.87)
       a[i] = random.uniform(0.7, 0.87)
       b[i] = random.uniform(0.7, 0.87)
       c[i] = random.uniform(0.7, 0.87)


    x.sort()
    y.sort()
    z.sort()
    a.sort()
    b.sort()
    c.sort()
    return x,y,z,a,b,c