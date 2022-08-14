"""
ARIMA forecast for the Yaounde weather dataset
"""
from math import sqrt
from numpy import split, array
import math
import pandas as pd
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


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
            actual[:, i], predicted[:, i])
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
ARIMA forecast
"""


def arima_forecast(history):
    """Convert history into a uni-variate series"""
    series = to_series(history)
    """Define the model"""
    model = ARIMA(series, order=(7, 0, 0))
    """Fit the model"""
    model_fit = model.fit()
    """Make forecast"""
    yhat = model_fit.predict(len(series), len(series)+23)
    return yhat


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
Define the names and functions for the models
to be evaluated
"""
models = dict()
models['arima'] = arima_forecast


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
