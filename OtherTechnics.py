# Using libraries to test clustering methods other than K-Means##########################################
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering

# General functions ##########################################
def calculate_and_print_scores(name, data, clusters):
    """
    Functions to calculate and print scores
    """
    silhouette_avg = silhouette_score(data, clusters)
    calinski_harabasz = calinski_harabasz_score(data, clusters)
    davies_bouldin = davies_bouldin_score(data, clusters)

    print(f"{name} - Silhouette Score: {silhouette_avg}")
    print(f"{name} - Calinski-Harabasz Index: {calinski_harabasz}")
    print(f"{name} - Davies-Bouldin Score: {davies_bouldin}")
    
    
    


# Load and process the dataset  ##########################################
Pdata = np.load('data.npy')
# Transposing the data to have samples as rows and features as columns
data = Pdata.T

print('Number of Samples', data.shape[0])
print('Number of Features', data.shape[1])

 #Doing all the diffents clustering  ##########################################

# DBSCAN Clustering 

#The eps(epsilon value) is first set to the lowest possible, then increased until the result are satisfying, for this we look at the three index
#the epsylon is the maximum distance between two points for them to be considered as part of the same neighborhood or cluster. 

#The min_sample is choosed arbitrairie depanding on what i have seen in other programs using it 
#Th emin_sample is the minimum number of points required to form a dense region

dbscan = DBSCAN(eps=100, min_samples=3)
dbscan.fit(data)
clusters_dbscan = dbscan.labels_

# Agglomerative Clustering 

# The n_clusters is taken form our previous results 
agglo = AgglomerativeClustering(n_clusters=5)
agglo.fit(data)
clusters_agglo = agglo.labels_

# Spectral Clustering 

# The n_clusters is taken form our previous results 
spectral = SpectralClustering(n_clusters=5, affinity='nearest_neighbors')
spectral.fit(data)
clusters_spectral = spectral.labels_

# Evaluate Clusters ##########################################
calculate_and_print_scores("DBSCAN", data, clusters_dbscan)
calculate_and_print_scores("Agglomerative", data, clusters_agglo)
calculate_and_print_scores("Spectral", data, clusters_spectral)
