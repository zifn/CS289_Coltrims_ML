import os
import shutil
import numpy as np

from .. import visualization

# Setup
np.random.seed(289)
test_data = np.random.normal(scale=1e6, size=(100, 15))

def test_cluster():
    """
    Checks the plotting of different clusters.
    """
    os.makedirs('tmp', exist_ok=True)

    clusters = np.repeat(np.arange(5), 20)
    fig1 = visualization.plot_electron_energy_vs_KER(test_data, clusters=clusters)

    fig1.savefig('tmp/electron-energy-vs-KER-clusters.png')

    assert os.path.exists('tmp/electron-energy-vs-KER-clusters.png')

    shutil.rmtree('tmp')

def test_electron_energy_vs_KER():
    """
    Checks the electron energy vs KER plotting function.
    """
    os.makedirs('tmp', exist_ok=True)

    fig1 = visualization.plot_electron_energy_vs_KER(test_data)
    fig2 = visualization.plot_electron_energy_vs_KER(test_data, bins=100)

    fig1.savefig('tmp/electron-energy-vs-KER.png')
    fig2.savefig('tmp/electron-energy-vs-KER-bins.png')

    assert os.path.exists('tmp/electron-energy-vs-KER.png')
    assert os.path.exists('tmp/electron-energy-vs-KER-bins.png')

    shutil.rmtree('tmp')

def test_electron_energies():
    """
    Checks the electron energies plotting function.
    """
    os.makedirs('tmp', exist_ok=True)

    fig1 = visualization.plot_electron_energies(test_data)
    fig2 = visualization.plot_electron_energies(test_data, bins=100)

    fig1.savefig('tmp/electron-energies.png')
    fig2.savefig('tmp/electron-energies-bins.png')

    assert os.path.exists('tmp/electron-energies.png')
    assert os.path.exists('tmp/electron-energies-bins.png')

    shutil.rmtree('tmp')

def test_ion_energies():
    """
    Checks the ion energies plotting function.
    """
    os.makedirs('tmp', exist_ok=True)

    fig1 = visualization.plot_ion_energies(test_data)
    fig2 = visualization.plot_ion_energies(test_data, bins=100)

    fig1.savefig('tmp/ion-energies.png')
    fig2.savefig('tmp/ion-energies-bins.png')

    assert os.path.exists('tmp/ion-energies.png')
    assert os.path.exists('tmp/ion-energies-bins.png')

    shutil.rmtree('tmp')

def test_KER_vs_angle():
    """
    Checks the KER vs angle plotting function.
    """
    os.makedirs('tmp', exist_ok=True)

    fig1 = visualization.plot_KER_vs_angle(test_data)
    fig2 = visualization.plot_KER_vs_angle(test_data, bins=100)

    fig1.savefig('tmp/KER-angle.png')
    fig2.savefig('tmp/KER-angle-bins.png')

    assert os.path.exists('tmp/KER-angle.png')
    assert os.path.exists('tmp/KER-angle-bins.png')

    shutil.rmtree('tmp')

def test_electron_energy_vs_ion_energy_difference():
    """
    Checks the electron energy vs ion energy difference plotting function.
    """
    os.makedirs('tmp', exist_ok=True)

    fig1 = visualization.plot_electron_energy_vs_ion_energy_difference(test_data)
    fig2 = visualization.plot_electron_energy_vs_ion_energy_difference(test_data, bins=100)

    fig1.savefig('tmp/electron-energy-ion-energy-difference.png')
    fig2.savefig('tmp/electron-energy-ion-energy-difference-bins.png')

    assert os.path.exists('tmp/electron-energy-ion-energy-difference.png')
    assert os.path.exists('tmp/electron-energy-ion-energy-difference-bins.png')

    shutil.rmtree('tmp')
