import torch

class EnergyDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list=[]):
        super().__init__()
        self.smiles_list = smiles_list
        self._data = {}

    def __len__(self):
        return len(self.smiles_list)

    def _prepare(self, idx):
        from .utils import (
            get_molecule_from_smiles,
            get_simulation_from_molecule,
        )
        smiles = self.smiles_list[idx]
        molecule = get_molecule_from_smiles(smiles)
        elements = torch.tensor([atom.element.atomic_number for atom in molecule.atoms])
        simulation = get_simulation_from_molecule(molecule)
        self._data[idx] = elements, simulation
        return elements, simulation

    def __getitem__(self, idx):
        assert isinstance(idx, int), "Only int is allowed."
        if idx in self._data:
            return self._data[idx]
        else:
            return self._prepare(idx)
