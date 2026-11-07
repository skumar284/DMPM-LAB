import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.array([
    [1, 2], [2, 3], [3, 2],
    [8, 8], [9, 9], [8, 9],
    [4, 10], [5, 11], [6, 10]
])

model = KMeans(n_clusters=3, random_state=42, n_init=10)

model.fit(X)

labels = model.labels_
centers = model.cluster_centers_

print("Cluster Labels:", labels)
print("Cluster Centers:")
print(centers)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker="X", s=200)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-Means Clustering")
plt.show()

"""# **K-MEDOID**"""

pip install scikit-learn-extra

import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids

X = np.array([
    [1, 2], [2, 3], [3, 2],
    [8, 8], [9, 9], [8, 9],
    [4, 10], [5, 11], [6, 10]
])

model = KMedoids(n_clusters=3, random_state=42)

model.fit(X)

labels = model.labels_
medoids = model.cluster_centers_

print("Cluster Labels:", labels)
print("Medoids:")
print(medoids)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(medoids[:, 0], medoids[:, 1], marker="X", s=200)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-Medoids Clustering")
plt.show()
