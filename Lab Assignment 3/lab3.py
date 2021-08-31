"""
    Name - Gajraj Singh Chouhan
    Lab Assignment 3
    Roll No - B19130
    Mobile No - +91-9351159849
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import decomposition

# import
data = pd.read_csv('landslide_data3.csv') # loading
data = data.drop(columns=['dates', 'stationid'])

def q1(PRINT=True):

    def fix_outliers():
        outliers=0
        medians = data.median()
        quartiles = data.quantile([0.25, 0.5, 0.75])
        for col in data.columns:
            q1, median, q3 = quartiles.loc[:, col].to_list()
            check_ifnot_outlier = lambda num : (q1 - (1.5 * (q3 - q1))) < num < (q3 + (1.5 * (q3 - q1)))
            for ind, x in enumerate(data.loc[:, col]):
                if not check_ifnot_outlier(x):
                    data.loc[ind, col] = medians[col]
                    outliers += 1
        return outliers
    outliers = fix_outliers()

    new_min, new_max = 3, 9
    data_ = data.copy()
    before_normalize = pd.concat([data_.min(), data_.max(),], axis=1, keys=["min", "max",]).round(decimals=5)
    for column in data_.columns:
        old_min = before_normalize.loc[column, "min"]
        old_max = before_normalize.loc[column, "max"]
        min_max = lambda val : (((val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min 
        data_[column] = data_[column].apply(min_max)

    after_normalize = pd.concat([data_.min(), data_.max(),], axis=1, keys=["min", "max",]).round(decimals=5)
    before_standardize = pd.concat([data.mean(), np.std(data)], axis=1, keys=["mean", "std-dev",]).round(decimals=5)
    for column in data.columns:
        mean = data[column].mean()
        standard_dev = np.std(data[column])
        normalize = lambda val : (val - mean) / (standard_dev)
        data[column] = data[column].apply(normalize)
    after_standardize = pd.concat([data.mean(), np.std(data)], axis=1, keys=["mean", "std-dev",]).round(decimals=5)
    
    if PRINT :
        print('\nQ1 : \n')
        print(f'outliers = {outliers}')                        
        print('Before Normalization - ')
        print(before_normalize)
        print('After Normalization - ')
        print(after_normalize)
        print('Before Standardization - ')
        print(before_standardize)
        print('After Standardization - ')
        print(after_standardize)

    return data # for q3

def q2(no_plot=False):
    print('\nQ2 : \n')

    total_sample = 1000
    mean = [0, 0]
    N = len(mean)
    cov_matrix = np.array([
                            [6.84806467, 7.63444163],
                            [7.63444163, 13.02074623]
                        ])
    eig_val, eig_vec = np.linalg.eig(cov_matrix)
    data_mat = np.random.multivariate_normal(
                    mean=mean,
                    cov=cov_matrix,
                    size=total_sample,
                    ) # ..... <- 1000
                      # ..... <- 1000 => 2x1000 (width x height) matrix
    x1, x2 = np.split(data_mat, N, axis=1)
    e1, e2 = np.split(eig_vec, N, axis=1)
    # Question 2b

    e1_projection = [] # projections of both of these vectors on the data
    e2_projection = []

    for point in range(total_sample):
        projection = []
        for vec in (e1, e2):
            projection.append(vec * np.vdot(vec, np.array([x1[point], x2[point]]))) # finding projection by dot product
        e1_projection.append(projection[0])
        e2_projection.append(projection[1])

    if no_plot is True:
        reconstruction = []
        for point1, point2 in zip(e1_projection, e2_projection):
            reconstruction.append(np.array(point1 + point2))
        reconstruction = np.array(reconstruction)
        reconstruction = reconstruction.reshape((1000, 2))
        print(f'error = {mean_squared_error(data_mat, reconstruction, squared=False)}')

        # One method using PCA also
        pca = decomposition.PCA(n_components=2)
        pca.fit(data_mat)
        pro = pca.transform(data_mat)
        pro = pca.inverse_transform(pro)
        comp = data_mat.flatten()
        pro = pro.flatten()
        ans = 0
        for i in range(len(comp)):
            ans += (comp[i] - pro[i])**2
        ans /= len(comp)
        ans = ans**0.5
        print(pca.explained_variance_ratio_)
        print(pca.explained_variance_)
        print(ans)

        return
    
    print(f"eigen-values-1 = {eig_val[0]}\neigen-values-2 = {eig_val[1]}")
    print(f"eigen-vector-1 = {e1}\neigen-vector-2 = {e2}")
    def plot():
        fig, ax = plt.subplots()
        ax.scatter(x1, x2, color='#5d00ff', marker='x')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_xlim(-22, 20)
        ax.set_ylim(-14, 15)

        ax.set_title('Scatter Plot of 2D Synthetic Data of 1000 samples')
        fig.savefig('Q2-scatter.png')
        
        for vec in (e1, e2):
            x_dir, y_dir = vec.flatten()
            ax.quiver(  # plotting the quiver
                0,0, x_dir, y_dir,
                width=0.004,
                color='#e0305f',
                scale=(8 if y_dir > 0 else 4)
                )

        return fig, ax

    fig, ax = plot()

    # Question 2a
    ax.set_title('Plot of 2D Synthetic Data and eigen directions')
    fig.savefig('Q2a.png')
    plt.close(fig)

    for points in e1_projection:
        ax.scatter(
            *points,
            color='#ff00ff',
            marker='x'
            )
    ax.set_title('Projecting the points on eigenvector 1')
    fig.savefig('Q2b1.png')
    plt.close()

    fig, ax = plot()

    for points in e2_projection:
        ax.scatter(
            *points,
            color='#ff00ff',
            marker='x'
            )
    ax.set_title('Projecting the points on eigenvector 2')
    fig.savefig('Q2b2.png')
    plt.close()

    return e1_projection, e2_projection

def q3():
    print('\nQ3 : \n')    
    data = q1(PRINT=0)

    pca = decomposition.PCA(n_components=2)
    pca.fit(data)
    principalComponents = pca.transform(data)
    data = data.values # data frame to np array
    corr_matrix = np.dot(np.transpose(data), data) # correlation matrix of the data
    val, vec = np.linalg.eig(corr_matrix)  # eigen values and eigen vectors
    print("Variance along the eigen vectors (in %)", 100*pca.explained_variance_ratio_)
    print("Variance along the eigen vectors (in value)", pca.explained_variance_)
    print("Eigenvalues of those vectors", val[:2])
    
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
    plt.title('Scatter Plot after Dimension Reduction')
    plt.xlabel('Column 1')
    plt.ylabel('Column 2')
    plt.savefig('Q3-scatter.png')
    plt.close()
    
    val = sorted(list(val), reverse=1)
    cout = list(range(1, 8))
    plt.bar(cout, val, color='red', width=0.4)
    plt.xlabel('Index')
    plt.ylabel('Magnitude')
    plt.title('Eigenvalues')
    plt.savefig('Q3-eigenvalues.png')
    plt.close()

    def find_rmse(a):
        pca = decomposition.PCA(n_components=a)
        pca.fit(data)
        pro = pca.transform(data)
        pro = pca.inverse_transform(pro)
        comp = data.flatten()
        pro = pro.flatten()
        ans = 0
        for i in range(len(comp)):
            ans += (comp[i] - pro[i])**2
        ans /= len(comp)
        ans = ans**0.5
        return ans
    
    rmse = [find_rmse(i) for i in range(1, 8)]
    plt.scatter(cout, rmse)
    plt.xlabel('values of l')
    plt.ylabel('rmse error value')
    plt.title('rmse error value corresponding to l values')
    plt.savefig('Q3-rmse.png')
    plt.close()

q1()
q2()
q2(no_plot=True)
q3()