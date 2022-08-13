"""
1D-CNN model for multi-step T2M forecasting
"""
import math
from math import sqrt
import matplotlib.pyplot as plt
from numpy import array, split
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.metrics import mean_squared_error
from keras.layers.convolutional import Conv1D, MaxPooling1D


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
Convert history into inputs and outputs
"""


def to_supervised(train, n_input, n_out=24):
    X, y = list(), list()
    in_start = 0
    """
    Step over the entire history, 
    one time step at a time
    """
    for _ in range(len(train)):
        """
        Define the end of the input sequence
        """
        in_end = in_start + n_input
        out_end = in_end + n_out
        """
        Ensure that there is enough 
        data for this instance
        """
        if out_end < len(train):
            x_input = train[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(train[in_end:out_end, 0])
        """
        Move along one time step
        """
        in_start += 1
    # print(array(X).shape)
    # print(array(y).shape)
    return array(X), array(y)


"""
Train the model
"""


def build_model(train, n_input):
    """
    Prepare data
    """
    train_x, train_y = to_supervised(train, n_input)
    """
    Define parameters
    """
    verbose, epochs, batch_size = 0, 20, 4
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    """
    Define model
    """
    model = Sequential()
    model.add(Conv1D(filters=16,
                     kernel_size=3,
                     activation='relu',
                     input_shape=(n_timesteps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    """
    Fit the network
    """
    model.fit(train_x,
              train_y,
              epochs=epochs,
              batch_size=batch_size,
              verbose=verbose)
    return model


"""
Make a forecast
"""


def forecast(model, history, n_input):
    """
    Flatten data
    """
    data = array(history)
    # shape_1 = data.shape[0]*data.shape[1]
    # data = data.reshape(shape_1, data.shape[2])
    """
    Retrieve last observations for input data
    """
    input_x = data[-n_input:, 0]
    """
    Reshape into [1, n_input, 1]
    """
    input_x = input_x.reshape((1, len(input_x), 1))
    """
    Forecast the next day
    """
    yhat = model.predict(input_x, verbose=0)
    """
    Only the vector forecast is needed
    """
    yhat=yhat[0]
    return yhat


"""
Evaluate a single model
"""


def evaluate_model(train, test, n_input):
    """
    Fit model
    """
    model = build_model(train, n_input)
    """
    History is a list of daily data
    """
    history = [x for x in train]
    """
    Walk forward validation over each day
    """
    predictions = list()
    for i in range(len(test)):
        """
        Predict the day
        """
        yhat_sequence = forecast(model, history, n_input)
        """
        Store the predictions
        """
        predictions.append(yhat_sequence)
        """
        Get real observation and 
        add to history for predicting the next week
        """
        history.append(test[i, :])
    """
    Evaluate predictions hours for each day
    """
    predictions = array(predictions)
    score, scores = evaluate_forecasts(
        test[:, :], predictions)

    return score, scores


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
Evaluate model and get scores
"""
n_input = 4
score, scores = evaluate_model(train, test, n_input)


"""
Summarize scores
"""
summarize_scores('cnn', score, scores)


"""
Plot scores
"""
hours = ['0', '1', '2', '3', '4', '5', '6', '7', '8',
         '9', '10', '11', '12', '13', '14', '15', '16',
         '17', '18', '19', '20', '21', '22', '23']

plt.plot(hours, scores, marker='o', label='cnn')
plt.show()
