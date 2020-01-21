import pandas as pd
import numpy as np

from src import configs


def do_preprocessing(debug=False, save=True):
    train = pd.read_csv(configs.train, index_col='id')
    test = pd.read_csv(configs.test, index_col='id')
    structures = pd.read_csv(configs.structures).set_index('molecule_name')

    if debug:
        train = train.sample(frac=.01, random_state=42)
        test = test.sample(frac=.01, random_state=42)

    target = train.pop('scalar_coupling_constant')
    molecules = train.molecule_name.values
    train = get_representation(train, structures)
    test = get_representation(test, structures)

    if save:
        if debug:
            np.save(configs.train_fin_debug, train)
            np.save(configs.test_fin_debug, test)
            np.save(configs.target_fin_debug, target)
            np.save(configs.molecules_fin_debug, molecules)
        else:
            np.save(configs.train_fin, train)
            np.save(configs.test_fin, test)
            np.save(configs.target_fin, target)
            np.save(configs.molecules_fin, molecules)

    return train, test, target, molecules


def load_preprocessed(debug=False):  # returns train, test, target, molecules
    if debug:
        return np.load(configs.train_fin_debug), np.load(configs.test_fin_debug), np.load(configs.target_fin_debug), \
               np.load(configs.molecules_fin_debug, allow_pickle=True)
    else:
        return np.load(configs.train_fin), np.load(configs.test_fin), np.load(configs.target_fin), \
               np.load(configs.molecules_fin, allow_pickle=True)


def get_representation(df, structures):
    new_df = np.zeros((len(df), 29 * 4))  # (number of samples, max atoms in molecule x number of features)

    for index, sample in enumerate(df.iterrows()):
        sample_df = np.zeros((29, 4))

        # retrieve the molecule name for a particular bond
        data = sample[1]  # iterrows returns tuple of (index, data)
        mol = data[['molecule_name']].values[0]

        # get x y z coordinates of all the atoms in the given molecule as np array
        mol_structure = structures.loc[mol]
        num_atoms = len(mol_structure)
        mol_structure = mol_structure[['x', 'y', 'z']].values

        # get coordinates of the two atoms in the bond
        atom1 = mol_structure[data[['atom_index_0']].values[0]]
        atom2 = mol_structure[data[['atom_index_1']].values[0]]

        # calculate center by taking the mean of the coordinates of the atoms in the bond
        bond_center = np.divide(np.add(atom1, atom2), 2)
        mol_structure = np.subtract(mol_structure, bond_center)
        mol_structure = np.hstack((mol_structure, np.ones((num_atoms, 1))))  # indicator that these rows correspond to an atom not padding
        sample_df[:num_atoms] = mol_structure

        sample_df = sample_df.flatten()
        new_df[index] = sample_df

    new_df = np.reshape(new_df, (len(df), 29, 4))

    return new_df
