# Regression examples

## [regression.py](#regression.py)
Predicting the median price of homes in a given Boston suburb in the mid-70's, given certain data points.

### Data:
* 506 data point split into 404 training samples, 102 test samples.
* Features have different scale.
* Prices in USD thousands.

### Model:
* Keras Sequential with two Dense layers
* K-fold cross validation

### Results:

<img width="400" src="/Images/Validation_MAE_by_epoch.png" />
