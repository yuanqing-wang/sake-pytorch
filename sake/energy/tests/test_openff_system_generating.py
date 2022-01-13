import pytest

def test_import():
    from sake.energy import utils
    from sake.energy.utils import get_forcefield

def test_caffeine_system():
    from sake.energy.utils import get_molecule_from_smiles, get_simulation_from_molecule
    molecule = get_molecule_from_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    simulation = get_simulation_from_molecule(molecule)

def test_single_energy():
    from sake.energy.utils import get_molecule_from_smiles, get_simulation_from_molecule, single_energy_evaluation
    molecule = get_molecule_from_smiles("C")
    simulation = get_simulation_from_molecule(molecule)
    import numpy as np
    x = np.random.randn(5, 3)
    energy = single_energy_evaluation(x, simulation=simulation)
