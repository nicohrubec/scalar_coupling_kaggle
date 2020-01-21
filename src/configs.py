from pathlib import Path

project_folder = Path.cwd().parent
data_folder = project_folder / 'data'
models_folder = project_folder / 'models'

train = data_folder / 'train.csv'
structures = data_folder / 'structures.csv'
test = data_folder / 'test.csv'

train_fin = data_folder / 'train_fin.npy'
train_fin_debug = data_folder / 'train_fin_debug.npy'
test_fin = data_folder / 'test_fin.npy'
test_fin_debug = data_folder / 'test_fin_debug.npy'
target_fin = data_folder / 'target_fin.npy'
target_fin_debug = data_folder / 'target_fin_debug.npy'
