from src import preprocessing
from src import training

debug = True  # if true then load small subset of the data
preprocess = False  # whether to run preprocessing or load numpy arrays from disk

if preprocess:
    train, test, target, molecules, types = preprocessing.do_preprocessing(debug=debug)
else:
    train, test, target, molecules, types = preprocessing.load_preprocessed(debug=debug)

training.train_KFolds(train, test, target, molecules, types, n_folds=5, debug=debug)


