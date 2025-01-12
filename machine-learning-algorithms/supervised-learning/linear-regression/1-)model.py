import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics




# Problem: Predict the price of a house based on given features

# Read csv file
def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    return df

def get_all_features(df):
    #get all columns
    #all_columns = df.columns
    return df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population', 'Price']]

def get_predict_column(df):
    return df['Price']



if __name__ == "__main__":
    
    #step 0 to load the data
    df = read_csv_file('data_sets/USA_Housing.csv')

    #Step 1 to get all features
    x_features = get_all_features(df) 

    #Step 2 to get the predict column which is y value
    y_predict = get_predict_column(df)

    #Step 3 to split the data into training set  and testing set, here is y value is the actual value
    x_train, x_test, y_train, y_test = train_test_split(x_features, y_predict, test_size=0.4, random_state=101)

    # Step 4 to create linear regression model
    linear_regression_model = LinearRegression()

    # Step 5 fit the model with training set
    linear_regression_model.fit(x_train, y_train)

    # Step 6 cofficient and intercept
    coefficient = linear_regression_model.coef_
    coefficient_data_frame = pd.DataFrame(coefficient, x_features.columns, columns=['Coefficient'])
    print(coefficient_data_frame)

    # Step 7 predict the model, we want to set value which model never seen before
    prediction = linear_regression_model.predict(x_test)

    # Step 8 to plot the prediction and actual value
    # oerfect straight line is the best fit line
    plt.scatter(y_test, prediction)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.show()

    # steo 9 observe residual
    # if you have normal disturbuted residual then your model is correct choice for the data
    # if you have non normal disturbuted residual then linear regression model is not correct choice for the data
    sns.distplot((y_test - prediction))
    plt.show()

    # Step 10 to show linear regression model performance
    print(metrics.mean_squared_error(y_test, prediction))
    print(metrics.mean_absolute_error(y_test, prediction))
    print(metrics.explained_variance_score(y_test, prediction))

   
