"""
Performing pair-wise correlations
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


"""
Load and process dataset
"""
df_1 = pd.read_csv('ydata.csv')  # Load dataset
del df_1['YEAR']    # Delete YEAR column
n = 23
df = df_1.iloc[:-n]  # remove last 23 records
# print(df.columns)


"""
Compute pairwise correlations
"""


def corr():
    for i in df.columns:
        correlations = df.corr()
        print(' ')
        print(' ')
        print(f'> Correlations for {i}  ')
        print(correlations[i])


# corr()


"""
Visualize Pairwise Correlations
"""


def draw_corr():
    correlations = df.corr()
    sns.heatmap(correlations)
    plt.show()


draw_corr()
