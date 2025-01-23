from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

if __name__ == '__main__':

    # create artificial data
    data = make_blobs(n_samples=100, n_features=2, centers=4, cluster_std=1.8, random_state=101)

    plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='rainbow')
    plt.show()

    #create model
    kmeans = KMeans(n_clusters=4, random_state=101)

    #train the data
    kmeans.fit(data[0])

    # so you can not compare actually but this case we have results, because we create artificial data
    print(kmeans.cluster_centers_)
    print(kmeans.labels_)


    # if you have virtual data then you can compare it.
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))
    ax1.set_title('K-Means Clustering')
    ax1.scatter(data[0][:, 0], data[0][:, 1], c=kmeans.labels_, cmap='rainbow')

    ax2.set_title('Original Data set')
    ax2.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='rainbow')
    plt.show()





