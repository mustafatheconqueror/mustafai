import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier


if __name__ == "__main__":

    #read data
    df = pd.read_csv('data_sets/kyphosis.csv')
    print(df.head())

    #visualize data
    sns.pairplot(df, hue='Kyphosis', diag_kind='hist')

    #prepare data
    x_features = df.drop('Kyphosis', axis=1)
    y_feature = df['Kyphosis']

    x_train, x_test, y_train, y_test = train_test_split(x_features, y_feature, test_size=0.3, random_state=101)

    #create model, decision tree model
    model = DecisionTreeClassifier()

    #train the model
    model.fit(x_train, y_train)

    #predictions.
    predictions = model.predict(x_test)

    # see the results
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(accuracy_score(y_test, predictions))


    # Random forest model
    random_forest_model = RandomForestClassifier(n_estimators=200)

    random_forest_model.fit(x_train, y_train)
    random_forest_predictions = random_forest_model.predict(x_test)



