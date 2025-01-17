import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def choose_best_k(x_train, y_train):
    for i in range(1,100):
        knn_model = KNeighborsClassifier(n_neighbors=i)
        knn_model.fit(x_train, y_train)
        predictions = knn_model.predict(x_test)
        print(f"K: {i}, Accuracy: {knn_model.score(x_test, y_test)}")


def plot_error_rate(x_train, y_train, error_rate):
    plt.figure(figsize=(10,6))
    plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
            markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')

if __name__ == "__main__":

    # get data
    df = pd.read_csv('data_set/Classified Data',index_col=0)

    # KNN kullanırkan standart scaling transformation yapmak güzel bir fikir olacaktır.
    scaler = StandardScaler()
    scaler.fit(df.drop('TARGET CLASS',axis=1))
    scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))  

    # as is datamı standtize edip düzenlemiş olduk.
    df_x_features = pd.DataFrame(scaled_features, columns=df.columns[:-1])
    df_y = df['TARGET CLASS']

    # split data into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(df_x_features,df_y, test_size=0.3, random_state=101 )

    # train model
    knn_model = KNeighborsClassifier(n_neighbors=1)
    knn_model.fit(x_train, y_train)

    # predict
    predictions = knn_model.predict(x_test)

    # see the results and metrics
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions ))


