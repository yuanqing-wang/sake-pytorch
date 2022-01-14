import pytest

def test_import():
    from sake.energy.dataset import EnergyDataset

def test_dataset():
    from sake.energy.dataset import EnergyDataset
    dataset = EnergyDataset(["C", "CC"])
    elements, simulation = dataset[0]
