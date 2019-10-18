from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten
from loguru import logger


class Model:
    def __init__(self, train_data_shape):
        self.model = Sequential()
        self.train_data_shape = train_data_shape

    def build(self):
        self.model.add(LSTM(100, input_shape=self.train_data_shape))
        self.model.add(Dropout(rate=0.2))
        self.model.add(Dense(1, activation='linear'))

        self.model.compile(loss='mse', optimizer='adam')
        logger.info('Model compiled.')

    def train(self, x, y, epochs, batch_size):
        logger.info('Training model.')
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size)
        logger.info('Model trained.')
