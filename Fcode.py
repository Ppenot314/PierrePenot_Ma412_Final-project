import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
from sklearn.model_selection import train_test_split

## General function ##############################################################


def initialize_centroids(data, k):
    """
    Randomly selects k data points as initial centroids
    """
    indices = np.random.choice(data.shape[0], size=k, replace=False)
    return data[indices]

def assign_clusters(data, centroids):
    """
    Assigns each data point to the nearest centroid
    """
    clusters = []
    for point in data:
        distances = np.sqrt(np.sum((point - centroids) ** 2, axis=1))
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

def update_centroids(data, clusters, k):
    """
    Recalculates centroids as the mean of all data points assigned to each cluster
    """
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        points_in_cluster = data[clusters == i]
        if len(points_in_cluster) > 0:
            new_centroids[i] = np.mean(points_in_cluster, axis=0)
    return new_centroids

def k_means(data, k, max_iterations=100):
    """
    Performs K-Means clustering
    """
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids


## 1 Load the dataset ##############################################################

data = np.load('data.npy')
# Transposing the data to have samples as rows and features as columns
data = data.T