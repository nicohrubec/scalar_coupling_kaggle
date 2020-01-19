from pathlib import Path

project_folder = Path.cwd().parent
data_folder = project_folder / 'data'
models_folder = project_folder / 'models'

train = data_folder / 'train.csv'
structures = data_folder / 'structures.csv'
test = data_folder / 'test.csv'
