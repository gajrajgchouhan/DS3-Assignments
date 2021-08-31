import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
from scipy import spatial as spatial

def plot_clusters(true_data, preds, cluster_center, cluster_name, savefig="", title=""):

    """
        This function plots the clusters from the data
        
        :true_data: Dataset, This dataframe provides the x and y data for the scatter plot
        :preds: These are our predicted clusters from each point in dataset
        :cluster_center: This contains cluster centers for each cluster
        :cluster_name: This contains the name of cluster (e.g. 0, 1, 2, ....)
        :savefig: For saving the plot as a png
        :title: Title of the plot
    """

    colors = plt.cm.get_cmap('hsv', len(cluster_name)+1) # get colors for each cluster using get_cmap. This will give us len(cluster_name) colors in a object form.
    
    for i, c in enumerate(cluster_name): # iterate through each cluster name
        if c == -1: # -1 is given by DBScan for noise
            clrs = 'grey' # make it grey
            label = 'Noise' # label it 'Noise'
        else:
            clrs = colors(c) # get color for it
            label=f'Cluster {c}' # label it by its name
        df = true_data[preds == c] # get the points from dataset whose prediction was cluster `c`
        x, y = df.iloc[:, 0], df.iloc[:, 1] # x and y axis
        plt.scatter( # plotting the x and y axis
                x, y,
                label=label,
                color=clrs
            )
        if c != -1:
            plt.text(
                    cluster_center[i][0] + 0.03, cluster_center[i][1] + 0.1,
                    f"Cluster {i}",
                    weight='bold',
                    fontsize=9,
                )
    
    plt.scatter(
        cluster_center[:, 0], cluster_center[:, 1], # plotting the cluster centers
        s=250, marker='*',
        c='red', edgecolor='black',
        label='Centroids'
    )
    
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    if savefig != "" : plt.savefig(f"{savefig}.png")
    plt.show()
    plt.close()

def purity_score(label, pred):

    """
        This function calculates the purity score from prediction and original classes.

        :label: This contains the original classes from our data set
        :pred: This contains our predicted clusters

        Logic: We will group all points which are in a cluster then find the purity score by adding which class is occuring most in that cluster and then dividing by total number by elements.
    """
    
    df = pd.concat([label, pd.DataFrame(pred)], axis=1)
    df.set_axis(['label', 'pred'], axis=1, inplace=True)
    
    s = 0

    for x, cluster in df.groupby('pred'):
        s += cluster['label'].value_counts().iloc[0] # adding the most occuring class in a cluster

    return s / label.shape[0]

def C_Center(data, pred):

    """
        This functions finds the cluster centers.

        :data:  Our dataset
        :pred:  Our predictions

        Logic: We will group all points which are in a cluster then find the cluster center from the mean of all the points (x and y co ordinates).
    """
    
    d = pd.concat([data, pd.DataFrame(pred)], axis=1)
    d.set_axis([*d.columns[:-1], 'pred'], axis=1, inplace=True)

    if -1 in d['pred'].values:
        return d.groupby('pred').mean().drop(index=-1).to_numpy()
    else:
        return d.groupby('pred').mean().to_numpy()

def dbscan_predict(dbscan_model, X_new, metric=spatial.distance.euclidean):

    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 

    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_):
            if metric(x_new, x_core) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break
    return y_new