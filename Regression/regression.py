# Predicting the median price of homes in a given Boston suburb in the mid-70's, given certain data points.
# Data:
# * 506 data point split into 404 training samples, 102 test samples.
# * Features have different scale.
# Prices in USD thousands.

import keras
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt


def load_data():
    """

    Load Keras dataset "Boston Housing Data".

    Columns (features):
    1. Per capita crime rate.
    2. Proportion of residential land zoned for lots over 25,000 square feet.
    3. Proportion of non-retail business acres per town.
    4. Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
    5. Nitric oxides concentration (parts per 10 million).
    6. Average number of rooms per dwelling.
    7- Proportion of owner-occupied units built prior to 1940.
    8. Weighted distances to five Boston employment centres.
    9. Index of accessibility to radial highways.
    10. Full-value property-tax rate per $10,000.
    11. Pupil-teacher ratio by town.
    12. 1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
    13. % lower status of the population.
    :return:
    """
    (train_d, train_t), (test_d, test_t) = keras.datasets.boston_housing.load_data()
    logger.info('Shape of data: ' + str(train_d.shape))
    return train_d, train_t, test_d, test_t


def normalize_data(train_d, test_d):
    """

    Feature-wise normalize (subtract the mean and divide by the standard deviation)
    :param train_d:
    :param test_d:
    :return: None.
    """
    # Gives a mean of 0 and standard deviation of 1 per feature.
    # Mean and standard deviation comes from the training data set.
    mean = train_d.mean(axis=0)
    std = train_d.std(axis=0)
    train_d -= mean
    train_d /= std

    test_d -= mean
    test_d /= std


def build_model():
    """

    Define the model with layers and functions.
    :return: Keras compiled model.
    """
    # Build network.
    # Small data set --> we use a small network so we don't overfit too much.
    # Scalar regression with unconstrained activation function (the network can learn to predict values in any range).
    model = keras.models.Sequential()
    # Two layers
    model.add(keras.layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(keras.layers.Dense(64, activation='relu'))
    # Single output (scalar).
    model.add(keras.layers.Dense(1))

    # Configuring the learning process.
    # Optimizer: rmsprop (divide the gradient by a running average of its recent magnitude)
    # Loss function:  mse (mean squared error)
    # Metrics: mae (mean absolute error), the absolute value of the difference between the predictions and the targets.
    #   Example: MAE = 0.5 means that predictions are off by 0.5 * 1000 = 500 USD.
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def k_fold_cross_valid(k: int, num_epochs: int, batch_size: int, train_data, test_data):
    """

    Train the model.

    We have a small sample and splitting data into training and validation sets are therefore not optimal.
    We use K-fold cross-validation instead:
        * Split data into K partitions
        * Instantiate K identical models and train eac on K-1 partitions
        * Evaluate on last partition.
    :param k: Number of partitions.
    :param num_epochs: Number of epochs to train the model.
    :param batch_size: Number of samples per gradient update.
    :param train_data: Our training data.
    :param test_data: Our test data.
    :return: All MAEs for graph.
    """
    num_val_samples = len(train_data) // k
    all_mae_histories = []

    for i in range(k):
        # Normalize data within the current fold instead of globally.
        normalize_data(train_data, test_data)

        logger.info('Processing fold #' + str(i))
        # Validation data for partition k.
        valid_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        valid_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

        # Training data for partition i
        partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0)

        model = build_model()
        # Train model.
        history = model.fit(partial_train_data, partial_train_targets, validation_data=(valid_data, valid_targets),
                            epochs=num_epochs, batch_size=batch_size, verbose=0)
        mae_history = history.history['val_mean_absolute_error']
        all_mae_histories.append(mae_history)

    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
    return average_mae_history


def plot_valid_scores(avg_maes: np.array):
    """

    Plot average MAE for every epoch.
    :param avg_maes: Array of MAE:s.
    :return: None
    """
    plt.plot(range(1, len(avg_maes) + 1), avg_maes)
    plt.xlabel('Epochs')
    plt.ylabel('Validation score: Mean Absolute Error (MAE)')
    plt.show()


if __name__ == '__main__':
    train_data, train_targets, test_data, test_targets = load_data()

    k = 4
    num_epochs = 500
    batch_size = 1
    avg_maes = k_fold_cross_valid(k, num_epochs, batch_size, train_data, test_data)

    plot_valid_scores(avg_maes)

    # New, compiled model
    model = build_model()
    model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
    test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
    logger.info('MAE score:' + str(test_mae_score))
