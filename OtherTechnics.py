#in this part we will use librairys to test quickly other methode than the Kmeans
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering


## General function ##############################################################



def calculate_and_print_scores(name, data, clusters):
    """
    
    Function to calculate and print scores
    """
    silhouette_avg = silhouette_score(data, clusters)
    calinski_harabasz = calinski_harabasz_score(data, clusters)
    davies_bouldin = davies_bouldin_score(data, clusters)

    print(f"{name} - Silhouette Score: {silhouette_avg}")
    print(f"{name} - Calinski-Harabasz Index: {calinski_harabasz}")
    print(f"{name} - Davies-Bouldin Score: {davies_bouldin}")


## 1 Load and process the dataset ######################################################
data = np.load('data.npy')
# Transposing the data to have samples as rows and features as columns
data = data.T

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2)



# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(train_data)
train_clusters_dbscan = dbscan.labels_

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=5)
agglo.fit(train_data)
train_clusters_agglo = agglo.labels_

# Spectral Clustering
spectral = SpectralClustering(n_clusters=5, affinity='nearest_neighbors')
spectral.fit(train_data)
train_clusters_spectral = spectral.labels_

# Evaluate Clusters
calculate_and_print_scores("DBSCAN", train_data, train_clusters_dbscan)
calculate_and_print_scores("Agglomerative", train_data, train_clusters_agglo)
calculate_and_print_scores("Spectral", train_data, train_clusters_spectral)


