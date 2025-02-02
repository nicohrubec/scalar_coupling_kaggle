import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from numpy import linalg

from src import configs


def do_preprocessing(debug=False, save=True):
    train = pd.read_csv(configs.train, index_col='id')
    test = pd.read_csv(configs.test, index_col='id')
    structures = pd.read_csv(configs.structures).set_index('molecule_name')

    if debug:
        train = train.sample(frac=.001, random_state=42)
        test = test.sample(frac=.001, random_state=42)

    target = train.pop('scalar_coupling_constant')
    molecules = train.molecule_name.values
    types = train.type.values

    train, test, structures = encode_categories(train, test, structures)

    train = get_representation(train, structures)
    test = get_representation(test, structures)

    if save:
        if debug:
            np.save(configs.train_fin_debug, train)
            np.save(configs.test_fin_debug, test)
            np.save(configs.target_fin_debug, target)
            np.save(configs.molecules_fin_debug, molecules)
            np.save(configs.types_fin_debug, types)
        else:
            np.save(configs.train_fin, train)
            np.save(configs.test_fin, test)
            np.save(configs.target_fin, target)
            np.save(configs.molecules_fin, molecules)
            np.save(configs.types_fin, types)

    return train, test, target, molecules, types


def encode_categories(train, test, structures):
    traintest = pd.concat([train, test], axis=0)

    # encode atom type and coupling type as integers
    le = LabelEncoder()
    structures['atom'] = le.fit_transform(structures['atom'])
    traintest['type'] = le.fit_transform(traintest['type'])

    train = traintest[:len(train)]
    test = traintest[len(train):]

    return train, test, structures


def load_preprocessed(debug=False):  # returns train, test, target, molecules
    if debug:
        return np.load(configs.train_fin_debug), np.load(configs.test_fin_debug), np.load(configs.target_fin_debug), \
               np.load(configs.molecules_fin_debug, allow_pickle=True), \
               np.load(configs.types_fin_debug, allow_pickle=True)
    else:
        return np.load(configs.train_fin), np.load(configs.test_fin), np.load(configs.target_fin), \
               np.load(configs.molecules_fin, allow_pickle=True), \
               np.load(configs.types_fin, allow_pickle=True)


def get_representation(df, structures):
    new_df = np.zeros((len(df), 29 * 9))  # (number of samples, max atoms in molecule x number of features)

    for index, sample in enumerate(df.iterrows()):
        sample_df = np.zeros((29, 9))

        # retrieve the molecule name for a particular bond
        data = sample[1]  # iterrows returns tuple of (index, data)
        mol = data[['molecule_name']].values[0]

        # get x y z coordinates of all the atoms in the given molecule as np array
        mol_structure = structures.loc[mol]
        num_atoms = len(mol_structure)
        atom_types = mol_structure[['atom']].values
        coupling_type = data[['type']].values[0]
        mol_structure = mol_structure[['x', 'y', 'z']].values

        # get coordinates of the two atoms in the bond
        atom1 = mol_structure[data[['atom_index_0']].values[0]]
        atom2 = mol_structure[data[['atom_index_1']].values[0]]

        # calculate center by taking the mean of the coordinates of the atoms in the bond
        bond_center = np.divide(np.add(atom1, atom2), 2)
        # repeat bond center vector for each atom in the molecule
        bond_center = np.tile(bond_center, (num_atoms, 1))
        mol_structure_centered = np.subtract(mol_structure, bond_center)
        # distances to bond center
        dists_center = np.reshape(linalg.norm(mol_structure, axis=1), (num_atoms, 1))
        # dists to atoms in the bond
        atom1 = np.tile(atom1, (num_atoms, 1))
        atom1_dists = np.reshape(linalg.norm(np.subtract(mol_structure, atom1), axis=1), (num_atoms, 1))
        atom2 = np.tile(atom2, (num_atoms, 1))
        atom2_dists = np.reshape(linalg.norm(np.subtract(mol_structure, atom2), axis=1), (num_atoms, 1))

        mol_structure = np.hstack((mol_structure_centered, dists_center, atom1_dists, atom2_dists,
                                   np.ones((num_atoms, 1)) * coupling_type, atom_types,
                                   np.ones((num_atoms, 1))))  # indicator that these rows correspond to an atom not padding
        sample_df[:num_atoms] = mol_structure

        sample_df = sample_df.flatten()
        new_df[index] = sample_df

    new_df = np.reshape(new_df, (len(df), 29, 9))

    return new_df
