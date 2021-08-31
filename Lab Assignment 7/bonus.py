"""
    Name - Gajraj Singh Chouhan
    Roll No - B19130
    Lab Assignment 7
    Mobile No - +91-9351159849
"""

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.style as style
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from functions import plot_clusters, purity_score, C_Center

style.use('seaborn')
matplotlib.rcParams['font.sans-serif'] = "DejaVu Sans"
matplotlib.rcParams['font.family'] = "sans-serif"

train_data = pd.read_csv("mnist-tsne-train.csv")
label_train = train_data['labels']
test_data = pd.read_csv("mnist-tsne-test.csv")
label_test = test_data['labels']            # splitting the labels and loading the data
train_data.drop(columns=['labels'], inplace=True)
test_data.drop(columns=['labels'], inplace=True)

dbscan_score = [] # matrix for purity score of Part B
score = {"KMeans_Train" : [], "GMM_Train" : []} # for storing distortion of Part A
index = (2, 5, 8, 12, 18, 20) # K for Part A

# K-means
for K in index:
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(train_data)
    score['KMeans_Train'].append(kmeans.inertia_) # `inertia_` is the distortion of the KMeans for a given K

# GMM
for K in index:
    gmm = GaussianMixture(n_components=K)
    gmm.fit(train_data)
    score['GMM_Train'].append(gmm.score(train_data)) # `score` is the log likelihood in GMM for K

# DBSCAN
for eps in (1, 5, 10):
    row = []
    for min_samples in (1, 10, 30, 50):
        dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_model.fit(train_data)
        DBSCAN_predictions = dbscan_model.labels_
        row.append(purity_score(label_train, DBSCAN_predictions)) # appending purity score
    dbscan_score.append(row)

score = pd.DataFrame.from_dict(score)
score.index = index
score.index.name = 'K'

# Part A
print('\nPart A\n')
print('Distortions : \n', score)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
# ax2 = ax1.twinx()

sns.lineplot(data=score['KMeans_Train'], marker='o', ax=ax1, color='red', label="Kmeans")

sns.lineplot(data=score['GMM_Train'], marker='o', ax=ax2, label="GMM")

plt.figtext(0.5, 0.04, 'K', ha='center', fontsize="large")
plt.figtext(0, 0.5, 'Distortions', va='center', rotation='vertical', fontsize="large")
fig.suptitle("Elbow Method", fontsize="large")
plt.legend(loc="best")
plt.xticks(index)
plt.savefig("Bonus-PartA.png")
plt.show()
plt.close()

# Part B
print('\nPart B\n')

dbscan_score = pd.DataFrame(np.array(dbscan_score))
dbscan_score.columns = (1, 10, 30, 50)
dbscan_score.index = (1, 5, 10)
dbscan_score.index.name = "eps"

print('\t\tmin_samples')
print(dbscan_score.to_string())

