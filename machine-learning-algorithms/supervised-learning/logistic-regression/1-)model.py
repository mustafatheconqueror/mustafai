import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
import cufflinks as cf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


#problem statement: Titanic - Machine Learning from Disaster

# imputa age according to passengeer class
def impute_age(cols):
    age = cols[0]
    p_class = cols[1]

    if pd.isnull(age):
        if p_class == 1:
            return 37
        elif p_class == 2:
            return 29
        else:
            return 24
    else:
        return age
    

if __name__ == "__main__":
    
    #step 1    get the train data frame
    train_df = pd.read_csv('data_sets/titanic_train.csv')
    
        
    # step 2 is actualy understrand the data, 
    # we will always have some missing values in the data set
    #plot the heatmap to see the missing values
    ## --> sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')


    # first, u have to analyze the data, and if that is classification problem,
    # most likely u want to see the distribution of the target variable
    #  --> sns.countplot(x='Survived', hue='Sex', data=train_df)
    
   # --> distirbution plot 
   #  sns.displot(train_df['Age'].dropna(), bins=30, kde=False)

    # --> cufflinks example
    #cf.go_offline()
    #train_df['Age'].iplot(kind='hist', bins=30, color='aliceblue')

    #step 3 : clean the data which is our ml model can handle
    # according to our missing datas, age is we can fill mean value
    # this is called imputation

    # impute age column, if not too much missing values
    train_df['Age'] = train_df[['Age', 'Pclass']].apply(impute_age, axis=1)
    
    # check sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False)
    #plot.show()

    # if missing values are too much, we can drop the column
    train_df.drop('Cabin', axis=1, inplace=True)

    #check is correct
    #print(train_df.columns)

    # if missing valu is too low like 10% or less, we can drop the row
    train_df.dropna(inplace=True)
    #check is correct
    sns.heatmap(train_df.isnull(), yticklabels=False, cbar= False)
    #plot.show()

    #step 4: convert categorical data to numerical data, called dummy variables.
    # ml algorithms can not handle categorical data

    #create dummy variables for sex column, and drop the first column
    sex = pd.get_dummies(train_df['Sex'], drop_first=True)
    embark = pd.get_dummies(train_df['Embarked'], drop_first=True)

    #concatenate the new columns to the original data frame
    train_df = pd.concat([train_df, sex, embark], axis=1)

    #check is correct
    print(train_df.head())

    #categorical datayı numeric dataya dönüştürdükten sonra, yeni colonları ekledik
    #şimdi de gereksiz colonları kaldırıyoruz + string data type olan colonları da  kaldırıyoruz
    train_df.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

    #check is correct, Ml algoritma için bütün dataları numerical data yapmamız müq oluyor.
    print(train_df.head())

    #Step 5 split data into training and testing data
    y = train_df['Survived']
    
    # Correctly drop the 'Survived' column and assign to x_features
    x_features = train_df.drop('Survived', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x_features, y, test_size=0.3, random_state=101)
    #split the data into training and testing data

    #step 6: train model
    logistic_regression_model = LogisticRegression()

    #step 7: fit model means give the data to the model, == train the model
    logistic_regression_model.fit(x_train, y_train)


    #step 8: predict the model, we want to set value which model never seen before
    prediction = logistic_regression_model.predict(x_test)

    #step 9: evaluate the model, and metrics
    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))

