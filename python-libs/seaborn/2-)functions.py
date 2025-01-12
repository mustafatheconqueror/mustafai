import pandas as pd
import seaborn as sns

def read_csv_file_with_pandas(file_path):
    df = pd.read_csv(file_path)
    return df

# sns.pairplot(df) = it gives us the pair plot of the dataframe
def use_pair_plot(df):
    print(sns.pairplot(df))

# sns.distplot(df[column_name]) = it gives us the distirbution plot of the specific column
def distirbution_plot_of_specific_column(df, column_name):
    print(sns.distplot(df[column_name]))

# sns.heatmap(df.corr()) = it gives us the heatmap of the correlation of the dataframe
def use_heatmap_function(df):
    print(sns.heatmap(df.corr()))

if __name__ == "__main__":
    df = read_csv_file_with_pandas('../../machine-learning-algorithms/supervised-learning/linear-regression/data_sets/USA_Housing.csv')
    #use_pair_plot(df)
    #distirbution_plot_of_specific_column(df, 'Price')
    use_heatmap_function(df)
    