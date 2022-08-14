"""
Visualizing the datasets using line plots
"""
import pandas as pd
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
Create line plot per variable
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
    Plot data
    """
    plt.plot(df[name])
    """
    Set title
    """
    plt.title(name, y=0)
    """
    Turn off ticks to remove clutter
    """
    plt.yticks([])
    plt.xticks([])
plt.show()
