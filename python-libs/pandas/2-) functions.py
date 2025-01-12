import pandas as pd

# Read csv file
def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    return df

# df.head() = print first 5 rows of the dataframe
def use_head_function(df):
    print(df.head())

# df.info() = print information about the dataframe
def use_info_function(df):
    print(df.info())

# df.describe() = it gives us the summary of the dataframe some statistic infos. like, count, mean, std, min, 25%, 50%, 75%, max
def use_describe_function(df):
    print(df.describe())

# df.columns = print the columns of the dataframe
def use_columns_function(df):
    print(df.columns)


# df.corr() = it gives us the correlation of the dataframe
def use_correlation_function(df):
    print(df.corr())


if __name__ == "__main__":
    df = read_csv_file('../../machine-learning-algorithms/supervised-learning/linear-regression/data_sets/USA_Housing.csv')
    #use_head_function(df)
    #use_info_function(df)
    #use_describe_function(df)
    #use_columns_function(df)    
    use_correlation_function(df)
