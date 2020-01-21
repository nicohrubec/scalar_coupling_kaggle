from src import preprocessing

debug = True  # if true then load small subset of the data
preprocess = False  # whether to run preprocessing or load numpy arrays from disk

if preprocess:
    train, test, target = preprocessing.do_preprocessing(debug=debug)
else:
    train, test, target = preprocessing.load_preprocessed(debug=debug)


