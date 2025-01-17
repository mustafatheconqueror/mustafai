import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


if __name__ == "__main__":

    # get data
    knn_df = pd.read_csv('data_Set/knn_data.csv')
    print(knn_df.head())
    print(knn_df.columns)

    sns.pairplot(knn_df, hue='TARGET CLASS')
    #plot.show()

    #Standardize the features
    scaler = StandardScaler()
    scaler.fit(knn_df.drop('TARGET CLASS', axis=1))
    scaled_features = scaler.transform(knn_df.drop('TARGET CLASS', axis=1))

    #create new datafram with scaled features
    knn_df_features_x = pd.DataFrame(scaled_features, columns=knn_df.columns[:-1])
    knn_df_target_y =  knn_df['TARGET CLASS']

    # split data into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(knn_df_features_x, knn_df_target_y, test_size=0.3, random_state=101)

    # train model
    knn_model = KNeighborsClassifier(n_neighbors=1)
    knn_model.fit(x_train, y_train)

    # predictions
    predictions = knn_model.predict(x_test)

    # see the results and metrics
    print(classification_report(y_test, predictions))
