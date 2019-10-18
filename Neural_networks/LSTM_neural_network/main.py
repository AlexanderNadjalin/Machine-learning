from Neural_networks.LSTM_neural_network.data_preperation import DataPrep
from Neural_networks.LSTM_neural_network.model import Model
from keras.layers import Flatten
from loguru import logger


def main():
    data = DataPrep('WM.csv', train_pct=0.8)
    m = Model(data.X_train.shape)
    m.build()
    x, y = data.get_training_data()
    x = Flatten()(x)
    m.train(x, y, 2, 32)
    print(data.frame.head())


if __name__ == '__main__':
    logger.info('Started.')
    main()
    logger.info('Ended.')