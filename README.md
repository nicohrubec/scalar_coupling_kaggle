# scalar_coupling_kaggle

1D CNN based approach for the "Predicting Molecular Properties" kaggle competition.
https://www.kaggle.com/c/champs-scalar-coupling/overview

The task in this competiton is to predict the magnetic interaction (scalar coupling constant) between two atoms in a molecule
given the structure of the molecule (x,y,z coordinates of the atoms in the molecule), the atom types 
and the scalar coupling type.

## model training

1. Clone the repository.
2. Create two folders "data" and "models" under /kaggle_scalar_coupling.
3. Download the competition data from the link given above and put it in the data folder.
4. Run the program with the preprocess parameter in the main file set to True in order to preprocess the numpy arrays for the 
model training.
5. Run with preprocess parameter set to False in order to load the preprocessed arrays from disk and start the training.

Model training can be done in debug mode for faster model development which trains the model on a sampled subset of the data.
Initially this is set to .01 % of the data which means only around 4500 samples are used.


## model description

The most important aspect of this challenge is to represent or describe the molecule in a way that is independent of
the order of the atoms given in the molecule. (It should not matter if Oxygen is listed first or last in the representation
of a Water Molecule.)

In order to achieve that the solution implemented here works as follows:
1. For each bond that must be predicted take all the atoms in the bond and calculate basic features for each 
of them. (distance of the atom to the bond center, distance to atom1 in the bond, atom type, coordinates relative to the bond center...)
2. This results in an array with the shape (num_samples, num_features_per_atom * num_atoms_in_the_molecule). This array is then reshaped for the conv net so that it has shape (num_samples, num_features_per_atom, num_atom_in_the_molecule). The maximum number of atoms in a molecule is 29, if a molecule has less than 29 atoms the remaining entries are simply padded with 0 so the shape of the array is then 
(num_samples, num_features_per_atom, 29).
3. A 1D CNN is then applied using the features as channels for the convolutions. Kernel size of the conv layers is 1 in order to satisfy
order invariance of the representation. Mean Pooling is then applied along the axis of the atoms before a couple of dense layers regress the scalar coupling constant for the given bond.


## results

Unfortunately, due to my limited local ressources, I couldn't yet run the net on the full dataset for long enough in order to report competitive results but it seems to work quite promising on a subset of the data. Also similar architectures have worked quite well in the competition.
