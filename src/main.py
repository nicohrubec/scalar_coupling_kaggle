from src import preprocessing

debug = True  # if true then load small subset of the data
preprocess = False  # whether to run preprocessing or load numpy arrays from disk

if preprocess:
    train, test, target, molecules = preprocessing.do_preprocessing(debug=debug)
else:
    train, test, target, molecules = preprocessing.load_preprocessed(debug=debug)


