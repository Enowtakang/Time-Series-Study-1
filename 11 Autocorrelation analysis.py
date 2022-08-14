"""
ACF and PACF plots of dataset
"""
from numpy import split
from numpy import array
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


"""
Split a univariate dataset into train/test sets
"""


def split_dataset(data):
    """
    Split into standard days
    """
    train, test = data[:52704], data[-8784:]
    """
    Restructure into windows of daily data
    """
    train = array(split(train, len(train)/24))
    test = array(split(test, len(test)/24))

    return train, test


"""
Convert windows of daily multivariate data 
into a series of uni-variate data
"""


def to_series(data):
    """Extract only the total from each day"""
    series = [day[:, 0] for day in data]
    """Flatten into a single series"""
    series = array(series).flatten()
    return series


"""
Load and process dataset
"""
df_1 = pd.read_csv('ydata.csv')  # Load dataset
del df_1['YEAR']    # Delete YEAR column
n = 23
df = df_1.iloc[:-n]  # remove last 23 records
# print(df.columns)


"""
Split into train and test sets
"""
train, test = split_dataset(df.values)

"""
Convert training data into series
"""
series = to_series(train)

"""
Plots
"""
plt.figure()
lags = 100

"""ACF"""
axis = plt.subplot(2, 1, 1)
plot_acf(series, ax=axis, lags=lags)

"""PACF"""
axis = plt.subplot(2, 1, 2)
plot_pacf(series, ax=axis, lags=lags)

plt.show()
