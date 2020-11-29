import numpy as np

import matplotlib.pyplot as plt

from .utils import (
    extract_data,
    kinetic_energy,
    ejection_angle,
    DEUTERON_MASS,
    ELECTRON_MASS,
    OXYGEN_MASS
)

def plot_data(x, y, clusters=None, colors=None, **kwargs):
    """
    Function to plot the COLTRIMS data. Takes two arrays of computed values and
    plots these along the x and y axes respectively. The function will color the
    points by cluster and include a legend if cluster labels are given.

    Parameters
    ------------
    x : array-like
        The x values to plot.
    y : array-like
        The y values to plot.
    clusters : array-like
        The cluster label for each (x, y) point, must be the same length as x
        and y.
    colors : array-like
        The colors assigned.
    kwargs
        Keyword arguments are passed to ax.plot().

    Returns
    --------
    tuple
        A tuple (fig, ax) containing the resulting matplotlib figure and axes.
    """
    fig, ax = plt.subplots()

    if colors is None:
        colors = [f'C{i}' for i in range(10)]

    if clusters is not None:
        assert len(clusters) == len(x)
        labels = np.unique(clusters)

        for i, label in enumerate(labels):
            c = colors[i%len(colors)]
            cluster_x, cluster_y = (x[clusters == label], y[clusters == label])
            ax.plot(cluster_x, cluster_y, '.', label=label, color=c, **kwargs)

        ax.legend()

    else:
        ax.plot(x, y, '.', color=colors[0], **kwargs)

    return fig, ax

def plot_2d_histogram(x, y, bins, **kwargs):
    """
    Function to plot a 2D histogram of the COLTRIMS data. Takes two arrays of
    computed values, bins them, and plots the 2D bins along the x and y axes as
    a 2D colormap.

    Parameters
    ------------
    x : array-like
        The x values to plot.
    y : array-like
        The y values to plot.
    bins : int or [int, int]
        If an int, the number of bins for the two dimensions. If [int, int],
        the number of bins along each axis (x, y).
    colorbar: bool
        If true, plots a colorbar.
    kwargs
        Keyword arguments are passed to ax.hist2d().

    Returns
    --------
    tuple
        A tuple (fig, ax) containing the resulting matplotlib figure and axes.
    """
    fig, ax = plt.subplots()
    ax.hist2d(x, y, bins=bins, **kwargs)

    return fig, ax

def format_plot(plot, xlabel='', ylabel='', title=''):
    """
    Function to format plots. Takes  The function will color the
    points by cluster and include a legend if cluster labels are given.

    Parameters
    ------------
    plot : tuple
        A tuple (fig, ax) containing the matplotlib figure and axes.
    xlabel : str
        The x-axis label.
    ylabel : str
        The y-axis label.
    title: str
        The plot title.

    Returns
    --------
    figure
        A matplotlib figure containing the resulting plot.
    """
    fig, ax = plot

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return fig

def plot_electron_energy_vs_KER(dataset, clusters=None, bins=None, **kwargs):
    """
    Function to plot the electron sum energy vs the total kinetic energy
    release, If bins is not None, the data will be binned along both axes.

    Parameters
    ------------
    dataset : array or dataframe
        A numpy array or dataframe containing the COLTRIMS data
    clusters : None or array
        The cluster labels for each entry of the dataset.
    bins : int or [int, int]
        If an int, the number of bins for the two dimensions. If [int, int],
        the number of bins along each axis (x, y).
    kwargs
        Keyword arguments are passed to ax.plot() or ax.hist2d(), depending on
        whether or not the data is binned.

    Returns
    --------
    figure
        A matplotlib figure containing the resulting plot.
    """
    particles = extract_data(dataset)
    masses = [DEUTERON_MASS] * 2 + [OXYGEN_MASS] + [ELECTRON_MASS] * 2

    energies = [kinetic_energy(*p.T, m) for p, m in zip(particles, masses)]

    electron_sum = sum(energies[-2:])
    KER = sum(energies)

    if bins:
        plot = plot_2d_histogram(electron_sum, KER, bins, **kwargs)
    else:
        plot = plot_data(electron_sum, KER, clusters, **kwargs)

    return format_plot(plot, 'Electron Sum Energy', 'KER')

def plot_electron_energies(dataset, clusters=None, bins=None, **kwargs):
    """
    Function to plot the energies of the first electron vs the second electron.
    If bins is not None, the data will be binned along both axes.

    Parameters
    ------------
    dataset : array or dataframe
        A numpy array or dataframe containing the COLTRIMS data
    clusters : None or array
        The cluster labels for each entry of the dataset.
    bins : int or [int, int]
        If an int, the number of bins for the two dimensions. If [int, int],
        the number of bins along each axis (x, y).
    kwargs
        Keyword arguments are passed to ax.plot() or ax.hist2d(), depending on
        whether or not the data is binned.

    Returns
    --------
    figure
        A matplotlib figure containing the resulting plot.
    """
    e1, e2 = extract_data(dataset)[-2:]

    energy1 = kinetic_energy(*e1.T, ELECTRON_MASS)
    energy2 = kinetic_energy(*e2.T, ELECTRON_MASS)

    if bins:
        plot = plot_2d_histogram(energy1, energy2, bins, **kwargs)
    else:
        plot = plot_data(energy1, energy2, clusters, **kwargs)

    return format_plot(plot, 'Electron 1 Energy', 'Electron 2 Energy')

def plot_ion_energies(dataset, clusters=None, bins=None, **kwargs):
    """
    Function to plot the energies of the first ion vs the second ion. If bins is
    not None, the data will be binned along both axes.

    Parameters
    ------------
    dataset : array or dataframe
        A numpy array or dataframe containing the COLTRIMS data
    clusters : None or array
        The cluster labels for each entry of the dataset.
    bins : int or [int, int]
        If an int, the number of bins for the two dimensions. If [int, int],
        the number of bins along each axis (x, y).
    kwargs
        Keyword arguments are passed to ax.plot() or ax.hist2d(), depending on
        whether or not the data is binned.

    Returns
    --------
    figure
        A matplotlib figure containing the resulting plot.
    """
    ion1, ion2 = extract_data(dataset)[:2]

    energy1 = kinetic_energy(*ion1.T, DEUTERON_MASS)
    energy2 = kinetic_energy(*ion2.T, DEUTERON_MASS)

    if bins:
        plot = plot_2d_histogram(energy1, energy2, bins, **kwargs)
    else:
        plot = plot_data(energy1, energy2, clusters, **kwargs)

    return format_plot(plot, 'Ion 1 Energy', 'Ion 2 Energy')

def plot_KER_vs_angle(dataset, clusters=None, bins=None, cos=False, **kwargs):
    """
    Function to plot the total kinetic energy release vs the angle between the
    two ionic fragments. If bins is not None, the data will be binned along both
    axes.

    Parameters
    ------------
    dataset : array or dataframe
        A numpy array or dataframe containing the COLTRIMS data
    clusters : None or array
        The cluster labels for each entry of the dataset.
    bins : int or [int, int]
        If an int, the number of bins for the two dimensions. If [int, int],
        the number of bins along each axis (x, y).
    cos : bool
        If true, plots the cosine of the angle along the y-axis. Otherwise,
        plots the angle in radians.
    kwargs
        Keyword arguments are passed to ax.plot() or ax.hist2d(), depending on
        whether or not the data is binned.

    Returns
    --------
    figure
        A matplotlib figure containing the resulting plot.
    """
    particles = extract_data(dataset)
    masses = [DEUTERON_MASS] * 2 + [OXYGEN_MASS] + [ELECTRON_MASS] * 2

    energies = [kinetic_energy(*p.T, m) for p, m in zip(particles, masses)]

    ion1, ion2 = particles[:2]

    KER = sum(energies)
    angles = ejection_angle(*ion1.T, *ion2.T, cos=cos)

    if bins:
        plot = plot_2d_histogram(KER, angles, bins, **kwargs)
    else:
        plot = plot_data(KER, angles, clusters, **kwargs)

    return format_plot(plot, 'KER', r'$\theta$')

def plot_KER_vs_electron_energy_difference(
    dataset,
    clusters=None,
    bins=None,
    **kwargs
):
    """
    Function to plot the total kinetic energy release vs the energy difference
    between the two electrons. If bins is not None, the data will be binned
    along both axes.

    Parameters
    ------------
    dataset : array or dataframe
        A numpy array or dataframe containing the COLTRIMS data
    clusters : None or array
        The cluster labels for each entry of the dataset.
    bins : int or [int, int]
        If an int, the number of bins for the two dimensions. If [int, int],
        the number of bins along each axis (x, y).
    kwargs
        Keyword arguments are passed to ax.plot() or ax.hist2d(), depending on
        whether or not the data is binned.

    Returns
    --------
    figure
        A matplotlib figure containing the resulting plot.
    """
    particles = extract_data(dataset)
    masses = [DEUTERON_MASS] * 2 + [OXYGEN_MASS] + [ELECTRON_MASS] * 2

    energies = [kinetic_energy(*p.T, m) for p, m in zip(particles, masses)]

    e1, e2 = particles[-2:]
    energy1 = kinetic_energy(*e1.T, ELECTRON_MASS)
    energy2 = kinetic_energy(*e2.T, ELECTRON_MASS)

    KER = sum(energies)
    energy_difference = np.abs(energy1 - energy2)

    if bins:
        plot = plot_2d_histogram(KER, energy_difference, bins, **kwargs)
    else:
        plot = plot_data(KER, energy_difference, clusters, **kwargs)

    return format_plot(plot, 'KER', 'Electron Energy Difference')
