"""
    Name - Gajraj Singh Chouhan
    Roll No - B19130
    Lab Assignment 7
    Mobile No - +91-9351159849
"""

import matplotlib
import pandas as pd
import matplotlib.style as style
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from functions import plot_clusters, purity_score, C_Center, dbscan_predict

style.use("seaborn")
matplotlib.rcParams["font.sans-serif"] = "DejaVu Sans"
matplotlib.rcParams["font.family"] = "sans-serif"

train_data = pd.read_csv("mnist-tsne-train.csv")
label_train = train_data["labels"]
test_data = pd.read_csv("mnist-tsne-test.csv")
label_test = test_data["labels"]  # splitting the labels and loading the data
train_data.drop(columns=["labels"], inplace=True)
test_data.drop(columns=["labels"], inplace=True)

# K-means
print("\nQuestion 1\n")

K = 10
kmeans = KMeans(n_clusters=K)  # KMeans
kmeans.fit(train_data)  # fitting

kmeans_prediction = kmeans.predict(train_data)  # predict training data
plot_clusters(
        train_data,
        kmeans_prediction,
        kmeans.cluster_centers_,
        range(K),
        savefig="Q1-Train",
        title="K-means Train Data",
    )
                        # plotting all the clusters from the predictions
print(f"Purity Score of training data = {purity_score(label_train, kmeans_prediction)}")
# purity score of clusters

kmeans_prediction = kmeans.predict(test_data)  # predict testing data
plot_clusters(
        test_data,
        kmeans_prediction,
        kmeans.cluster_centers_,
        range(K),
        savefig="Q1-Test",
        title="K-means Test Data",
    )
print(f"Purity Score of testing data = {purity_score(label_test, kmeans_prediction)}")

# GMM
print("\nQuestion 2\n")

K = 10
gmm = GaussianMixture(n_components=K)  # GMM
gmm.fit(train_data)

GMM_prediction = gmm.predict(train_data)
cluster_center = C_Center(train_data, GMM_prediction)  # cluster
plot_clusters(
        train_data,
        GMM_prediction,
        cluster_center,
        range(K),
        savefig="Q2-Train",
        title="GMM Train Data",
    )
print(f"Purity Score of training data = {purity_score(label_train, GMM_prediction)}")

GMM_prediction = gmm.predict(test_data)
cluster_center = C_Center(test_data, GMM_prediction)
plot_clusters(
        test_data,
        GMM_prediction,
        cluster_center,
        range(K),
        savefig="Q2-Test",
        title="GMM Test Data",
    )
print(f"Purity Score of testing data = {purity_score(label_test, GMM_prediction)}")


# DBSCAN
print("\nQuestion 3\n")

dbscan_model = DBSCAN(eps=5, min_samples=10)  # DBScan
dbscan_model.fit(train_data)

DBSCAN_predictions = dbscan_model.labels_
cluster_name = set(DBSCAN_predictions)
cluster_center = C_Center(train_data, DBSCAN_predictions)  # cluster center
plot_clusters(
        train_data,
        DBSCAN_predictions,
        cluster_center,
        cluster_name,
        savefig="Q3-Train",
        title="DBScan Train Data",
    )

print(f"Purity Score of training data = {purity_score(label_train, DBSCAN_predictions)}")

DBSCAN_predictions = dbscan_predict(dbscan_model, test_data.values)
cluster_name = set(DBSCAN_predictions)
cluster_center = C_Center(test_data, DBSCAN_predictions)
plot_clusters(
        test_data,
        DBSCAN_predictions,
        cluster_center,
        cluster_name,
        savefig="Q3-Test",
        title="DBScan Test Data",
    )

print(f"Purity Score of testing data = {purity_score(label_test, DBSCAN_predictions)}")
