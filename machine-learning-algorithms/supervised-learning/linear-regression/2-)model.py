import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#Problem Statement
"""
Linear Regression Project
Congratulations! You just got some contract work with an Ecommerce company based in New York City 
that sells clothing online but they also have in-store style and clothing advice sessions. 
Customers come in to the store, have sessions/meetings with a personal stylist, 
then they can go home and order either on a mobile app or website for the clothes they want.
The company is trying to decide whether to focus their efforts on their mobile app experience 
or their website. They've hired you on contract to help them figure it out! Let's get started!
Just follow the steps below to analyze the customer data (it's fake, don't worry I didn't give you 
real credit card numbers or emails)."""


def check_if_data_is_loaded_correctly(data_frame):
    print(data_frame.head())
    print(data_frame.info())
    print(data_frame.describe())
    #print(customers_df.columns) --> get the columns


if __name__ == "__main__":

    #step 1: read data
    customers_df = pd.read_csv('data_sets/ECommerce_Customer.csv')

    #step 2: check if data is loaded correctly
    #check_if_data_is_loaded_correctly(customers_df)

    #step 3: visualize data
    
    # Seaborn jointplot, x ve y tablosu yoğunluk haritasi gibi bir şey oluşturur.
    sns.jointplot(x= 'Time on Website', y= 'Yearly Amount Spent', data= customers_df)
    sns.jointplot(x= 'Time on App', y = 'Yearly Amount Spent', data= customers_df)

    # 2d hex plot
    sns.jointplot(x = 'Time on App', y = 'Length of Membership', data = customers_df, kind = 'hex')
    
    sns.pairplot(customers_df)
    
    # Matplotlib ile grafiği göster
    #plt.show()

    #step 3: get features and target

    x_features = customers_df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership', 'Yearly Amount Spent']]
    y_feature = customers_df['Yearly Amount Spent']


    #step 4: split data
    x_train, x_test, y_train, y_test = train_test_split(x_features, y_feature, test_size=0.3, random_state=101)

    #step 5: train model
    linear_regression_model = LinearRegression()

    #step 6: fit model means give the data to the model
    linear_regression_model.fit(x_train, y_train)

    #step 7: print the coefficients
    print(linear_regression_model.coef_)

    #step 8: predict the model
    predictions = linear_regression_model.predict(x_test)

    #step 9: show the results predictions and actual values 
    plt.scatter(y_test, predictions)
    plt.xlabel('Y Test')
    plt.ylabel('Predicted Y')
    plt.show()

    #step 10: evaluate the model and show the results
    metrics.mean_absolute_error(y_test, predictions)
    metrics.mean_squared_error(y_test, predictions)
    metrics.mean_squared_error(y_test, predictions)
    metrics.explained_variance_score(y_test, predictions)

    #step 11: show residuals, residuals are the difference between the actual values and the predicted values
    sns.distplot((y_test-predictions), bins=50)

    #step 12: show the coefficients
    coeffecients = pd.DataFrame(linear_regression_model.coef_, x_features.columns, columns=['Coeffecient'])
    coeffecients.columns = ['Coeffecient']
