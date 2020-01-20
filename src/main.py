from src import preprocessing

train, test, target, structures = preprocessing.do_preprocessing(debug=True)
train, test = preprocessing.do_engineering(train, test, structures)

