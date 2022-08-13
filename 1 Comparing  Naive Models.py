"""
Naive forecast strategies for the T2M
uni-variate time series dataset
"""
import math
from math import sqrt
from numpy import split
from numpy import array
import pandas as pd
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt


"""
Split a uni-variate dataset into train/test sets
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
Evaluating one or more daily forecasts
against expected values.
"""


def evaluate_forecasts(actual, predicted):
    scores = list()
    """
    Calculate an RMSE for each hour
    """
    for i in range(actual.shape[1]):
        """
        Calculate MSE
        """
        mse = mean_squared_error(
            actual[:, 1], predicted[:, i])
        """
        Calculate RMSE
        """
        rmse = sqrt(mse)
        """
        Store
        """
        scores.append(rmse)
    """
    Calculate the overall RMSE
    """
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            expression = actual[
                             row, col] - predicted[
                row, col]
            s += math.pow(expression, 2)
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))

    return score, scores


"""
Summarize scores
"""


def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


"""
Evaluate a single model
"""


def evaluate_model(model_func, train, test):
    """
    History is a list of daily data
    """
    history = [x for x in train]
    """
    Walk-forward validation over each day
    """
    predictions = list()
    for i in range(len(test)):
        """
        Predict the day
        """
        yhat_sequence = model_func(history)
        """
        Store the predictions
        """
        predictions.append(yhat_sequence)
        """
        Get real observations and add to history for 
        predicting the next week 
        """
        history.append(test[i, :])
    predictions = array(predictions)

    """
    Evaluate predictions hours for each day
    """
    score, scores = evaluate_forecasts(
        test[:, :], predictions)

    return score, scores


"""
Hourly persistence model
"""


def hourly_persistence(history):
    """
    Get the data for the previous day
    """
    last_day = history[-1]

    """
    Get the temperature for the last hour
    """
    value = last_day[-1]

    """
    Prepare 24 hour forecast
    """
    forecast = [value for _ in range(24)]

    return forecast


"""
Daily persistence model
"""


def daily_persistence(history):
    """
    Get the data for the prior day
    """
    last_day = history[-1]

    return last_day[:]


"""
Day one year ago persistence model
"""


def day_one_year_ago_persistence(history):
    """
    Get the data for the prior day
    """
    last_day = history[-366]

    return last_day[:]


"""
Splitting the dataset into standard days:
Load dataset and extract T2M
"""

df_1 = pd.read_csv('ydata.csv')  # Load dataset
n = 23
df_2 = df_1.iloc[:-n]  # remove last 23 records
df = df_2['T2M']    # Extract T2M

train, test = split_dataset(df.values)


"""
Define the names and functions for the models
to be evaluated
"""
models = dict()
models['hourly'] = hourly_persistence
models['daily'] = daily_persistence
models['day_oya'] = day_one_year_ago_persistence


"""
Evaluate each model
"""
hours = ['0', '1', '2', '3', '4', '5', '6', '7', '8',
         '9', '10', '11', '12', '13', '14', '15', '16',
         '17', '18', '19', '20', '21', '22', '23']

for name, func in models.items():
    """
    Evaluate and get scores
    """
    score, scores = evaluate_model(func, train, test)
    """
    Summarize scores
    """
    summarize_scores(name, score, scores)
    """
    Plot scores
    """
    plt.plot(hours, scores, marker='o', label=name)

"""
Show plot
"""
plt.legend()
plt.show()
