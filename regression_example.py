# Predicting the median price of homes in a given Boston suburb in the mid-70's, given certain data points.
# Data:
# * 506 data point split into 404 training samples, 102 test samples.
# * Features have different scale

import keras
import numpy as np

# Load data
(train_data, train_targets), (test_data, test_targets) = keras.datasets.boston_housing.load_data()
