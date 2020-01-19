import pandas as pd

from src import configs


def do_preprocessing():
    train = pd.read_csv(configs.train)
    test = pd.read_csv(configs.test)

    return train, test
