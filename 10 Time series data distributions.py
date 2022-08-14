"""
Visualizing time series data distributions
through histogram plots
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
Histogram plot for each variable
"""
plt.figure()
for i in range(len(df.columns)):
    """
    Create subplot
    """
    plt.subplot(len(df.columns), 1, i+1)
    """
    Get variable name
    """
    name = df.columns[i]
    """
    Create histogram
    """
    df[name].hist(bins=100)
    """
    Set title
    """
    plt.title(name, y=0, loc='right')
    """
    Turn off ticks to remove clutter
    """
    plt.yticks([])
    plt.xticks([])
plt.show()

