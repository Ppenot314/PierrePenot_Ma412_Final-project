#in this code we will focus on the visalisation of the clusters formed by the kmeans methode

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

def k_means(data, k, max_iterations=500):
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

## 2 vizualise the solution ########################################################

# Apply K-Means with a range of cluster counts
inertia = []
K_range = range(1, 20)
for k in K_range:
    kmeans = KMeans(n_clusters=k ).fit(data)
    inertia.append(kmeans.inertia_)

# Plotting the Graph
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, marker='o')
plt.title('Inertia in function of K using sklearn')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.xticks(K_range)
plt.grid(True)
plt.show()

## 3 visulise the same solution with our homemadecode ####################################


# Apply K-Means with a range of cluster counts
inertia = []
K_range = range(1, 20)
for k in K_range:
    clusters, centroids = k_means(data, k)
    #calculating inertia 
    k_inertia = 0 
    for i in range(k):
        cluster_data = data[clusters == i]
        cluster_inertia = np.sum((cluster_data - centroids[i])**2)
        k_inertia += cluster_inertia
    inertia.append(k_inertia)
    
# Plotting the Graph
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, marker='o')
plt.title('Inertia in function of K using homemade code')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.xticks(K_range)
plt.grid(True)
plt.show()