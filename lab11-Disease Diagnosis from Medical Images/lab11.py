import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
data = load_breast_cancer()

X = data.data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1])

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Breast Cancer Dataset")

plt.show()
kmeans = KMeans(n_clusters=2, random_state=42)

kmeans.fit(X_scaled)

labels = kmeans.labels_
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering on Breast Cancer Dataset")

plt.show()
