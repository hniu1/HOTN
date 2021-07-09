from sgt import SGT
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier, NearestNeighbors



def ReadSequentialData(InputFileName):
    RawTrajectories = []
    with open(InputFileName, 'rb') as f:
        list_raw = pickle.load(f)
    LoopCounter = 0
    for data in list_raw:
        ## [Ship1] [(Port1, time)] [(Port2, time)] [(Port3, time)]...
        ship = data[0]
        movements = [node[0] for node in data[1:]]
        timestamp = [node[1] for node in data[1:]]

        LoopCounter += 1
        if LoopCounter % 10000 == 0:
            print(LoopCounter)
        ## Test for movement length
        MinMovementLength = MinimumLengthForTraining + LastStepsHoldOutForTesting
        if len(movements) < MinMovementLength:
            continue

        RawTrajectories.append([ship, movements])

    return RawTrajectories

def SGTEmbedding(RawTrajectories):
    corpus = pd.DataFrame(RawTrajectories,
                          columns=['id', 'sequence'])

    sgt = SGT(kappa=1,
              flatten=True,
              lengthsensitive=False,
              mode='default')
    sgtembedding_df = sgt.fit_transform(corpus)

    return sgtembedding_df

def Clustering(sgtembedding_df):

    # Set the id column as the dataframe index
    sgtembedding_df = sgtembedding_df.set_index('id')
    pca = PCA(n_components=2)
    pca.fit(sgtembedding_df)

    X = pca.transform(sgtembedding_df)
    print(np.sum(pca.explained_variance_ratio_))
    df = pd.DataFrame(data=X, columns=['x1', 'x2'])
    df.head()

    kmeans = KMeans(n_clusters=1, max_iter=300)
    kmeans.fit(df)

    labels = kmeans.predict(df)
    centroids = kmeans.cluster_centers_
    distances = np.sqrt((X - centroids) ** 2)
    mean = np.mean(distances)
    std = np.std(distances)
    threshold = mean + 2*std

    fig = plt.figure(figsize=(5, 5))
    # colmap = {1: 'r', 2: 'g'}
    # colors = list(map(lambda x: colmap[x + 1], labels))
    plt.scatter(df['x1'], df['x2'], alpha=0.5)
    plt.show()
    print('check')

def AD_DBSCAN(sgtembedding_df):
    # Set the id column as the dataframe index
    sgtembedding_df = sgtembedding_df.set_index('id')
    pca = PCA(n_components=2)
    pca.fit(sgtembedding_df)

    X = pca.transform(sgtembedding_df)
    print(np.sum(pca.explained_variance_ratio_))
    df = pd.DataFrame(data=X, columns=['x1', 'x2'])
    df.head()

    outlier_detection = DBSCAN(eps=.1, metric='euclidean', min_samples = 2, n_jobs = -1)
    clusters = outlier_detection.fit_predict(df)

    colmap = {1: 'r', 0: 'g'}
    colors = list(map(lambda x: colmap[x], clusters))
    plt.scatter(df['x1'], df['x2'], color=colors, alpha=0.5, edgecolors=colors)
    plt.show()
    print('check')

def AD_KNN(sgtembedding_df):
    # Set the id column as the dataframe index
    sgtembedding_df = sgtembedding_df.set_index('id')
    pca = PCA(n_components=2)
    pca.fit(sgtembedding_df)

    X = pca.transform(sgtembedding_df)
    print(np.sum(pca.explained_variance_ratio_))
    df = pd.DataFrame(data=X, columns=['x1', 'x2'])
    df.head()
    n_neighbors_scaled = 0.3
    n_neighbors = int(round(n_neighbors_scaled * len(X)))
    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    neigh.fit(df)
    na_distance_0 = neigh.kneighbors(df)[0]
    lst_distances_0 = []
    for distances in na_distance_0:
        NN_distance = np.mean(distances)
        lst_distances_0.append(NN_distance)
    arr_dis_0 = np.asarray(lst_distances_0, dtype=np.float32)
    mean_0 = np.mean(arr_dis_0, axis=0)
    std_0 = np.std(arr_dis_0, axis=0)
    threshold_0 = mean_0 + 2 * std_0

    pred = []
    for id, distance in enumerate(lst_distances_0):
        if distance > threshold_0:
            label = 1
        else:
            label = 0
        pred.append(label)

    colmap = {1: 'r', 0: 'g'}
    colors = list(map(lambda x: colmap[x], pred))
    plt.scatter(df['x1'], df['x2'], color=colors, alpha=0.5, edgecolors=colors)
    plt.show()
    print('check')

MinimumLengthForTraining = 1
LastStepsHoldOutForTesting = 0
InputFolder = '../../data_preprocessed/tensor/Synthetic_1000/'
InputFileName = InputFolder + 'tensor_seq.pkl'

if __name__ == '__main__':
    RawTrajectories = ReadSequentialData(InputFileName)
    sgtembedding_df = SGTEmbedding(RawTrajectories)
    Clustering(sgtembedding_df)
    AD_DBSCAN(sgtembedding_df)
    AD_KNN(sgtembedding_df)
print('check')
