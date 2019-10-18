import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import features


class DataPrep:
    def __init__(self, file_name, train_pct):
        self.frame = self.csv_import(file_name)
        self.split = int(len(self.frame) * train_pct)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.set_features()
        self.set_train_test_data()
        self.normalise()

    def set_features(self):
        self.frame = features.add_fixed_features(self.frame)
        self.frame.dropna(inplace=True)

    def set_train_test_data(self):
        x = self.frame.iloc[:, 7:-1]
        y = self.frame.iloc[:, 5]
        self.X_train = x[:self.split]
        self.X_test = x[self.split:]
        self.y_train = y[:self.split]
        self.y_test = y[self.split:]

    def normalise(self):
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)

    def get_training_data(self):
        x = [self.X_train]
        y = [self.y_train]
        return np.array(x), np.array(y)

    def csv_import(self, file_name) -> pd.DataFrame:
        p = Path('yahoo_finance_data')
        df = pd.DataFrame()
        file_to_import = p.joinpath(file_name)
        try:
            df = pd.read_csv(file_to_import, sep=",", parse_dates=True, decimal='.')
            df['Date'] = pd.to_datetime(df['Date'])
            logger.info('Imported file: "' + str(file_name) + '".')
        except ImportError as err:
            logger.critical('Import of file "' + str(file_name) + '" failed with error:')
            logger.critical('  ' + str(err))
            logger.info('Aborted.')
            quit()
        return df
