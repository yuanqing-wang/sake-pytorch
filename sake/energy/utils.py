def get_dummy_integrator():
    from simtk import openmm, unit
    class DummyIntegrator(openmm.CustomIntegrator):

        """
        Ported from:
        https://openmmtools.readthedocs.io/en/0.17.0/_modules/openmmtools/integrators.html#DummyIntegrator

        Construct a dummy integrator that does nothing except update call the force updates.

        Returns
        -------
        integrator : mm.CustomIntegrator
            A dummy integrator.

        Examples
        --------

        Create a dummy integrator.

        >>> integrator = DummyIntegrator()

        """

        def __init__(self):
            timestep = 0.0 * unit.femtoseconds
            super(DummyIntegrator, self).__init__(timestep)
            self.addUpdateContextState()
            self.addConstrainPositions()
            self.addConstrainVelocities()
    return DummyIntegrator()

def get_forcefield(forcefield="openff_unconstrained-1.2.0.offxml"):
    from openff.toolkit.typing.engines.smirnoff import ForceField
    forcefield = ForceField(forcefield)
    return forcefield

def get_molecule_from_smiles(smiles):
    from openff.toolkit.topology import Molecule
    molecule = Molecule.from_smiles(smiles)
    return molecule

def get_simulation_from_molecule(molecule, forcefield=None):
    if forcefield is None:
        forcefield = get_forcefield()
    from openff.toolkit.topology import Topology
    topology = Topology.from_molecules(molecule)
    system = forcefield.create_openmm_system(topology)

    from simtk.openmm.app import Simulation
    integrator = get_dummy_integrator()
    simulation = Simulation(topology.to_openmm(), system, integrator)
    return simulation

def single_energy_evaluation(x, simulation):
    from simtk import unit
    simulation.context.setPositions(x)
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return energy

def tensor_energy_evaluation(x, simulation):
    import torch
    device = x.device
    dtype = x.dtype
    x = x.detach().cpu().numpy()
    energy = [single_energy_evaluation(_x, simulation=simulation) for _x in x]
    energy = torch.tensor(energy, device=device, dtype=dtype)
    return energy
