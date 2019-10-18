from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from loguru import logger


class Model:
    def __init__(self):
        self.model = Sequential()

    def build(self):
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(Dropout(rate=0.2))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(Dropout(rate=0.2))
        self.model.add(Dense(1, activation='linear'))

        self.model.compile(loss='mse', optimizer='adam')
        logger.info('Model compiled.')

    def train(self, x, y, epochs, batch_size):
        logger.info('Training model.')
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size)
        logger.info('Model trained.')
