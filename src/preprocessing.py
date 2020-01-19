import pandas as pd

from src import configs


def do_preprocessing(debug=False):
    train = pd.read_csv(configs.train, index_col='id')
    test = pd.read_csv(configs.test, index_col='id')

    if debug:
        train = train.sample(frac=.01, random_state=42)
        test = test.sample(frac=.01, random_state=42)

    return train, test
