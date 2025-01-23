import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report


def converter(private):
    if private == "Yes":
        return 1
    else:
        return 0


if __name__ == '__main__':
    # get the data
    df = pd.read_csv('data_sets/College_Data.csv', index_col=0)

    # some infos about data
    print(df.head())
    print(df.info())
    print(df.describe())

    # get some data visializations.
    # scatter plot
    sns.lmplot(x='Room.Board', y='Grad.Rate', data=df, hue='Private', fit_reg=False)

    g = sns.FacetGrid(df, hue='Private', palette='Blues', height=4, aspect=2)
    g = g.map(plt.hist, 'Outstate', bins=20, alpha=0.7)

    # create model
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(df.drop('Private', axis=1))

    # print some results
    print(kmeans.cluster_centers_)

    # evaluate
    df['Cluster'] = df['Private'].apply(converter)
    print(classification_report(df['Cluster'], kmeans.labels_))
    print(confusion_matrix(df['Cluster'], kmeans.labels_))