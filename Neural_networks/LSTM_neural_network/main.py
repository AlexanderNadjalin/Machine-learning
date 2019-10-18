from data_preperation import DataPrep
from model import Model
from loguru import logger


def main():
    data = DataPrep('WM.csv', train_pct=0.8)
    m = Model()
    m.build()
    x, y = data.get_training_data()
    m.train(x, y, 2, 32)
    print(data.frame.head())


if __name__ == '__main__':
    logger.info('Started.')
    main()
    logger.info('Ended.')