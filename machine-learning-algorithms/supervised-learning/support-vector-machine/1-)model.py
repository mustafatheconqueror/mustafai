from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import pandas as pd

if __name__ == "__main__":
    # create dataframe
    cancer = load_breast_cancer()

    #create data frame
    df = pd.DataFrame(data=cancer['data'], columns=cancer['feature_names'])


    # visualize to understand data set
    x_features = df
    y_feature = cancer['target']

    # train test and split data
    x_train, x_test, y_train, y_test = train_test_split(x_features, y_feature, test_size=0.2, random_state=42)

    #create model
    svm_model = SVC()

    # fit the model
    svm_model.fit(x_train, y_train)

    #predictions
    svm_model_predictions = svm_model.predict(x_test)

    # see the result
    print(classification_report(y_test, svm_model_predictions))
    print(confusion_matrix(y_test, svm_model_predictions))
    print(accuracy_score(y_test, svm_model_predictions))
