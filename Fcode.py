#in this code we will focus on judging the quality of the clusters created by our algorithm (reposing on the kmeans methode)
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

def silhouette_score(data, clusters, centroids):
    """
    Calculates the silhouette score for the current clustering
    """
    silhouette_scores = []
    for idx, point in enumerate(data):
        own_cluster = clusters[idx]
        in_cluster_distances = np.sqrt(np.sum((data[clusters == own_cluster] - point) ** 2, axis=1))
        nearest_cluster = np.argmin([np.mean(np.sqrt(np.sum((data[clusters == i] - point) ** 2, axis=1))) for i in range(len(centroids)) if i != own_cluster])
        out_cluster_distances = np.sqrt(np.sum((data[clusters == nearest_cluster] - point) ** 2, axis=1))
        score = (np.mean(out_cluster_distances) - np.mean(in_cluster_distances)) / max(np.mean(out_cluster_distances), np.mean(in_cluster_distances))
        silhouette_scores.append(score)
    return np.mean(silhouette_scores)

def k_means(data, k, max_iterations=200):
    """
    Performs K-Means clustering.
    """
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

def Optimal_k_finder(data, max_k, max_iterations=200):
    """
    Determines the optimal number of clusters (K) for K-Means clustering.
    """
    best_k = 2
    best_score = -1
    for k in range(2, max_k + 1):  # Starting from 2 because silhouette score is not defined for k=1
        clusters, centroids = k_means(data, k, max_iterations)
        score = silhouette_score(data, clusters, centroids)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k



## 1 Load the dataset ################################################################################

data = np.load('data.npy')
# Transposing the data to have samples as rows and features as columns
data = data.T


## 2 Find the optimal number of clusters #############################################################
max_k = 12  # Max number of clusters to consider, this value is sencitive 
optimal_k = Optimal_k_finder(data, max_k)
print("Max number of clusters to consider whe searching optimal_k:", max_k)
print("Optimal number of clusters:", optimal_k)

# Run K-Means with this optimal_k
clusters, centroids = k_means(data, optimal_k)


