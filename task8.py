import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import make_blobs
from sklearn.datasets import load_iris
import pandas as pd
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
def task1():
    X = np.array([[5,3],
    [10,15],
    [15,12],
    [24,10],
    [30,45],
    [85,70],
    [71,80],
    [60,78],
    [55,52],
    [80,91],])
    kmeans = KMeans(n_clusters=10, random_state=0)
    plt.scatter(X[:,0], X[:,1], s = 20)
    plt.show()
    kmeans.fit(X)
    y_kmeans = kmeans.fit_predict(X)
    print(y_kmeans)
def task2():
    digits = load_iris()
    kmeans = KMeans(n_clusters=5, random_state=0)
    clusters = kmeans.fit_predict(digits.data)
    print(clusters)

def task3():

    url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
    Bdata = pd.read_csv(url)
    Bdata.head()
    data = Bdata.iloc[:, 0:2].values
    plt.figure(figsize=(28, 12), dpi=180)
    dend = shc.dendrogram(shc.linkage(data, method='ward'))
    cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    y = cluster.fit_predict(data)
    print(y)
    plt.figure(figsize=(10, 7))
    plt.scatter(data[:, 0], data[:, 1], c=cluster.labels_, cmap='rainbow')
    plt.show()


print('Enter num task 1 or 2 or 3 or 4')
inp = int(input())
while inp != 0:
    if inp == 1:
        task1()
        break
    elif inp == 2:
        task2()
        break
    elif inp == 3:
        task3()
        break
    else:
        print("Check input")
        break


