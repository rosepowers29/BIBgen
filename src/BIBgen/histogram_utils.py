"""
Histogram generation module for BIB (Beam-Induced Background) analysis.

This module provides functions to create histograms from HDF5 data files
converted from SLCIO format, including time, energy, angular distributions,
and hit clustering analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from typing import Dict, List, Optional, Tuple, Union


def load_hit_data_from_hdf5(
    filepath: str,
    event_nums: Optional[List[int]] = None,
    collections: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Load hit data from HDF5 file and combine into arrays for analysis.
    
    Reads HDF5 file created by slcio_to_hdf5.py and extracts hit information
    from specified events and detector collections. Combines all hits into
    single arrays and computes derived quantities (phi, eta, r).
    
    Parameters
    ----------
    filepath : str
        Path to HDF5 file containing converted SLCIO data.
    event_nums : list of int, optional
        List of event numbers to load. If None, loads all events in file.
        Default is None.
    collections : list of str, optional
        List of collection names to load. If None, loads all available
        collections. Default is None.
    
    Returns
    -------
    data : dict of str to np.ndarray
        Dictionary containing combined hit data with keys:
        - 'time': hit times in ns
        - 'energy': energy deposition in GeV
        - 'x': x position in mm
        - 'y': y position in mm
        - 'z': z position in mm
        - 'r': radial distance from beam axis in mm (sqrt(x^2 + y^2))
        - 'phi': azimuthal angle in radians
        - 'eta': pseudorapidity
        - 'cell_id': cell identifier
    
    Raises
    ------
    FileNotFoundError
        If HDF5 file does not exist.
    KeyError
        If specified events or collections are not found in file.
    
    Examples
    --------
    >>> data = load_hit_data_from_hdf5('bib_data.hdf5')
    >>> print(f"Total hits: {len(data['time'])}")
    """
    import os
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")
    
    # Initialize lists to accumulate data
    all_times = []
    all_energy = []
    all_x = []
    all_y = []
    all_z = []
    all_cell_id = []
    
    with h5py.File(filepath, 'r') as f:
        # Get all event names if not specified
        if event_nums is None:
            event_keys = [k for k in f.keys() if k.startswith('evt_')]
        else:
            event_keys = [f'evt_{n}' for n in event_nums]
        
        # Loop through events
        for evt_key in event_keys:
            if evt_key not in f:
                continue
            
            evt_group = f[evt_key]
            
            # Loop through detector types (ECalColls, HCalColls, TrkHitColls)
            for det_type in evt_group.keys():
                det_group = evt_group[det_type]
                
                # Loop through collections
                for coll_name in det_group.keys():
                    # Skip if we're filtering collections
                    if collections is not None and coll_name not in collections:
                        continue
                    
                    coll_group = det_group[coll_name]
                    
                    # Check if datasets exist (some collections might be empty)
                    if 'hit_times' not in coll_group:
                        continue
                    
                    # Read datasets
                    times = coll_group['hit_times'][:]
                    x_pos = coll_group['hit_x_pos'][:]
                    y_pos = coll_group['hit_y_pos'][:]
                    z_pos = coll_group['hit_z_pos'][:]
                    cell_ids = coll_group['hit_cell_id'][:]
                    
                    # Get energy (different name for tracker vs calo)
                    if 'hit_energy' in coll_group:
                        energy = coll_group['hit_energy'][:]
                    elif 'hit_e_dep' in coll_group:
                        energy = coll_group['hit_e_dep'][:]
                    else:
                        energy = np.zeros(len(times))
                    
                    # Append to lists
                    all_times.extend(times)
                    all_energy.extend(energy)
                    all_x.extend(x_pos)
                    all_y.extend(y_pos)
                    all_z.extend(z_pos)
                    all_cell_id.extend(cell_ids)
    
    # Convert to numpy arrays
    times_arr = np.array(all_times)
    energy_arr = np.array(all_energy)
    x_arr = np.array(all_x)
    y_arr = np.array(all_y)
    z_arr = np.array(all_z)
    cell_id_arr = np.array(all_cell_id)
    
    # Calculate derived quantities
    r_arr = np.sqrt(x_arr**2 + y_arr**2)
    phi_arr = np.arctan2(y_arr, x_arr)
    
    # Calculate pseudorapidity: eta = -ln(tan(theta/2))
    theta = np.arctan2(r_arr, z_arr)
    # Avoid division by zero
    eta_arr = np.where(
        np.abs(np.tan(theta/2)) > 1e-10,
        -np.log(np.tan(theta/2)),
        np.sign(z_arr) * 10.0  # Large eta for very forward/backward
    )
    
    data = {
        'time': times_arr,
        'energy': energy_arr,
        'x': x_arr,
        'y': y_arr,
        'z': z_arr,
        'r': r_arr,
        'phi': phi_arr,
        'eta': eta_arr,
        'cell_id': cell_id_arr
    }
    
    return data


def plot_time_histogram(
    time_data: np.ndarray,
    bins: int = 100,
    time_range: Optional[Tuple[float, float]] = None,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create histogram of hit time distribution.
    
    Parameters
    ----------
    time_data : np.ndarray
        Array of hit times in nanoseconds.
    bins : int, optional
        Number of bins. Default is 100.
    time_range : tuple of float, optional
        (min, max) time range in ns. If None, uses data range.
        Default is None.
    output_path : str, optional
        Path to save figure. Default is None.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    
    Raises
    ------
    ValueError
        If time_data is empty.
    """
    if len(time_data) == 0:
        raise ValueError("time_data cannot be empty")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(time_data, bins=bins, range=time_range, 
            histtype='step', color='blue', linewidth=1.5)
    
    ax.set_xlabel('Hit Time [ns]', fontsize=12)
    ax.set_ylabel('Number of Hits', fontsize=12)
    ax.set_title('Hit Time Distribution', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_phi_histogram(
    phi_data: np.ndarray,
    bins: int = 100,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create histogram of azimuthal angle (phi) distribution.
    
    Parameters
    ----------
    phi_data : np.ndarray
        Array of phi values in radians (typically -pi to pi).
    bins : int, optional
        Number of bins. Default is 100.
    output_path : str, optional
        Path to save figure. Default is None.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    
    Raises
    ------
    ValueError
        If phi_data is empty.
    """
    if len(phi_data) == 0:
        raise ValueError("phi_data cannot be empty")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(phi_data, bins=bins, range=(-np.pi, np.pi),
            histtype='step', color='green', linewidth=1.5)
    
    ax.set_xlabel('φ [rad]', fontsize=12)
    ax.set_ylabel('Number of Hits', fontsize=12)
    ax.set_title('Azimuthal Angle Distribution', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_eta_histogram(
    eta_data: np.ndarray,
    bins: int = 100,
    eta_range: Tuple[float, float] = (-5, 5),
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create histogram of pseudorapidity (eta) distribution.
    
    Parameters
    ----------
    eta_data : np.ndarray
        Array of pseudorapidity values.
    bins : int, optional
        Number of bins. Default is 100.
    eta_range : tuple of float, optional
        (min, max) eta range. Default is (-5, 5).
    output_path : str, optional
        Path to save figure. Default is None.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    
    Raises
    ------
    ValueError
        If eta_data is empty.
    """
    if len(eta_data) == 0:
        raise ValueError("eta_data cannot be empty")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(eta_data, bins=bins, range=eta_range,
            histtype='step', color='purple', linewidth=1.5)
    
    ax.set_xlabel('η (Pseudorapidity)', fontsize=12)
    ax.set_ylabel('Number of Hits', fontsize=12)
    ax.set_title('Pseudorapidity Distribution', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_energy_histogram(
    energy_data: np.ndarray,
    bins: int = 100,
    log_scale: bool = True,
    energy_range: Optional[Tuple[float, float]] = None,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create histogram of energy deposition distribution.
    
    Parameters
    ----------
    energy_data : np.ndarray
        Array of energy deposition values in GeV.
    bins : int, optional
        Number of bins. Default is 100.
    log_scale : bool, optional
        If True, use logarithmic x-axis. Default is True.
    energy_range : tuple of float, optional
        (min, max) energy range. If None, uses data range.
        Default is None.
    output_path : str, optional
        Path to save figure. Default is None.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    
    Raises
    ------
    ValueError
        If energy_data is empty or contains non-positive values
        when log_scale is True.
    
    Examples
    --------
    >>> data = load_hit_data_from_hdf5('data.hdf5')
    >>> fig = plot_energy_histogram(data['energy'], log_scale=True)
    >>> plt.show()
    """
    if len(energy_data) == 0:
        raise ValueError("energy_data cannot be empty")
    
    if log_scale and np.any(energy_data <= 0):
        # Filter out zero/negative energies for log scale
        energy_data = energy_data[energy_data > 0]
        if len(energy_data) == 0:
            raise ValueError("No positive energy values for log scale")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if log_scale:
        log_energy = np.log10(energy_data)
        if energy_range is not None:
            plot_range = (np.log10(energy_range[0]), np.log10(energy_range[1]))
        else:
            plot_range = None
        
        ax.hist(log_energy, bins=bins, range=plot_range,
                histtype='step', color='red', linewidth=1.5)
        ax.set_xlabel('log₁₀(E_dep) [GeV]', fontsize=12)
    else:
        ax.hist(energy_data, bins=bins, range=energy_range,
                histtype='step', color='red', linewidth=1.5)
        ax.set_xlabel('E_dep [GeV]', fontsize=12)
    
    ax.set_ylabel('Number of Hits', fontsize=12)
    ax.set_title('Energy Deposition Distribution', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def compute_deltaR_clustering(
    eta: np.ndarray,
    phi: np.ndarray,
    deltaR_bins: np.ndarray = np.logspace(-3, 1, 50),
    weights: Optional[np.ndarray] = None,
    max_hits: Optional[int] = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute hit clustering profile in Delta R space.
    
    For each hit, counts the number of neighboring hits within various Delta R
    radii, where Delta R = sqrt((Δη)² + (Δφ)²). This characterizes spatial
    correlation and clustering of hits, which is crucial for understanding
    shower structure and hit patterns.
    
    Parameters
    ----------
    eta : np.ndarray
        Array of pseudorapidity values.
    phi : np.ndarray
        Array of azimuthal angles in radians.
    deltaR_bins : np.ndarray, optional
        Bin edges for Delta R histogram. Default is np.logspace(-3, 1, 50).
    weights : np.ndarray, optional
        Weights for each hit (e.g., energy). If provided, returns weighted
        counts. Default is None.
    max_hits : int, optional
        Maximum number of hits to use (for performance). If None or if
        len(eta) < max_hits, uses all hits. For large datasets, randomly
        samples hits. Default is 10000.
    
    Returns
    -------
    deltaR_centers : np.ndarray
        Centers of Delta R bins.
    avg_neighbors : np.ndarray
        Average number (or weighted sum) of neighboring hits within each
        Delta R bin.
    
    Raises
    ------
    ValueError
        If eta and phi have different lengths or are empty.
    
    Examples
    --------
    >>> data = load_hit_data_from_hdf5('data.hdf5')
    >>> deltaR, neighbors = compute_deltaR_clustering(
    ...     data['eta'],
    ...     data['phi'],
    ...     weights=data['energy']
    ... )
    >>> import matplotlib.pyplot as plt
    >>> plt.loglog(deltaR, neighbors, 'o-')
    >>> plt.xlabel('ΔR')
    >>> plt.ylabel('Avg # of Neighbors')
    >>> plt.show()
    
    Notes
    -----
    This is an O(N²) algorithm. For datasets with >10000 hits, the function
    will automatically subsample for performance.
    """
    if len(eta) != len(phi):
        raise ValueError(f"eta and phi must have same length: {len(eta)} vs {len(phi)}")
    
    if len(eta) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    # Subsample if too many hits
    n_hits = len(eta)
    if max_hits is not None and n_hits > max_hits:
        print(f"Warning: Subsampling {max_hits} hits from {n_hits} for performance")
        indices = np.random.choice(n_hits, max_hits, replace=False)
        eta = eta[indices]
        phi = phi[indices]
        if weights is not None:
            weights = weights[indices]
        n_hits = max_hits
    
    if weights is None:
        weights = np.ones(n_hits)
    
    n_bins = len(deltaR_bins) - 1
    neighbor_counts = np.zeros((n_hits, n_bins))
    
    # Compute pairwise Delta R
    print(f"Computing Delta R clustering for {n_hits} hits...")
    for i in range(n_hits):
        if i % 1000 == 0:
            print(f"  Processing hit {i}/{n_hits}")
        
        # Compute differences
        deta = eta - eta[i]
        dphi = phi - phi[i]
        
        # Handle phi wraparound
        dphi = np.abs(dphi)
        dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
        
        # Compute Delta R
        deltaR = np.sqrt(deta**2 + dphi**2)
        
        # Histogram neighbors
        for j in range(n_bins):
            mask = (deltaR > deltaR_bins[j]) & (deltaR <= deltaR_bins[j+1])
            neighbor_counts[i, j] = np.sum(weights[mask])
    
    # Average over all hits
    avg_neighbors = np.mean(neighbor_counts, axis=0)
    
    # Compute bin centers (geometric mean)
    deltaR_centers = np.sqrt(deltaR_bins[:-1] * deltaR_bins[1:])
    
    return deltaR_centers, avg_neighbors


def plot_deltaR_clustering(
    eta: np.ndarray,
    phi: np.ndarray,
    deltaR_bins: np.ndarray = np.logspace(-3, 1, 50),
    weights: Optional[np.ndarray] = None,
    max_hits: Optional[int] = 10000,
    output_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, plt.Figure]:
    """
    Create visualization of Delta R clustering analysis.
    
    Computes and plots the hit clustering profile showing how hits are
    spatially correlated. Shows average number of neighboring hits as a
    function of Delta R radius.
    
    Parameters
    ----------
    eta : np.ndarray
        Array of pseudorapidity values.
    phi : np.ndarray
        Array of azimuthal angles in radians.
    deltaR_bins : np.ndarray, optional
        Bin edges for Delta R. Default is np.logspace(-3, 1, 50).
    weights : np.ndarray, optional
        Optional weights (e.g., energy) for each hit. Default is None.
    max_hits : int, optional
        Maximum hits to use for performance. Default is 10000.
    output_path : str, optional
        Path to save figure. Default is None.
    
    Returns
    -------
    deltaR_centers : np.ndarray
        Centers of Delta R bins.
    avg_neighbors : np.ndarray
        Average neighboring hit count per bin.
    fig : matplotlib.figure.Figure
        The figure object.
    
    Raises
    ------
    ValueError
        If input validation fails.
    
    Examples
    --------
    >>> data = load_hit_data_from_hdf5('data.hdf5')
    >>> deltaR, neighbors, fig = plot_deltaR_clustering(
    ...     data['eta'],
    ...     data['phi'],
    ...     weights=data['energy'],
    ...     output_path='clustering.png'
    ... )
    >>> plt.show()
    """
    # Compute clustering
    deltaR_centers, avg_neighbors = compute_deltaR_clustering(
        eta, phi, deltaR_bins, weights, max_hits
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(deltaR_centers, avg_neighbors, 'o-', 
              linewidth=2, markersize=4, label='Data')
    
    ax.set_xlabel('ΔR', fontsize=12)
    
    if weights is not None:
        ax.set_ylabel('Avg Weighted Neighbors within ΔR', fontsize=12)
    else:
        ax.set_ylabel('Avg # of Neighbors within ΔR', fontsize=12)
    
    ax.set_title('Hit Clustering Profile', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    
    # Add reference line for random distribution (~ ΔR²)
    if len(avg_neighbors) > 0 and avg_neighbors[0] > 0:
        random_ref = avg_neighbors[0] * (deltaR_centers / deltaR_centers[0])**2
        ax.loglog(deltaR_centers, random_ref, '--', color='gray', 
                 alpha=0.5, label='Random (∝ΔR²)')
    
    ax.legend()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return deltaR_centers, avg_neighbors, fig
