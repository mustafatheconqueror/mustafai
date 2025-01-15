"""In this project we will be working with a fake advertising data set, 
indicating whether or not a particular internet user clicked on an Advertisement.
 We will try to create a model that will predict whether or not they will click on
   an ad based off
 the features of that user."""

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == "__main__":

    # read data
    ad_data = pd.read_csv('data_sets/advertising.csv')
    print(ad_data.head())
    print(ad_data.describe())
    print(ad_data.info())

    # explore data with plots and dygrams.
    #sns.distplot(ad_data['Age'], kde=False, bins=35)
    #plot.show()
    
    # alternative way to create histogram of the age
    #ad_data['Age'].hist(bins=40)

    print(ad_data.columns)
    #sns.jointplot(y='Area Income', x ='Age', data=ad_data)
    #plot.show()

    #sns.jointplot(y='Daily Time Spent on Site', x='Age', data=ad_data, kind='kde', color='red')
    #plot.show()

    
    #sns.jointplot(x = 'Daily Time Spent on Site', y = 'Daily Internet Usage', data=ad_data, color='green')

    #genel diagramı çiziyor bütün featurelara göre
    sns.pairplot(ad_data, hue='Clicked on Ad', palette='bwr')
    #plot.show()


    # get x features and y label
    y = ad_data['Clicked on Ad']
    x_features = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]

    #split datat into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(x_features, y, test_size=0.3, random_state=101)

    #create a model
    logistic_regression_model = LogisticRegression()

    # fit the model
    logistic_regression_model.fit(x_train, y_train)

    # predic the model
    predictions = logistic_regression_model.predict(x_test)

    #evaluate the result
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))