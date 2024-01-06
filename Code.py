import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
data = np.load('data.npy')

# Transposing the data to have samples as rows and features as columns
data = data.T

#step 1 vizualise the solution. 

# Apply K-Means with a range of cluster counts
inertia = []
K_range = range(1, 50)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    inertia.append(kmeans.inertia_)

# Plotting the Graph
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, marker='o')
plt.title('Inertia in function of K')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.xticks(K_range)
plt.grid(True)
plt.show()