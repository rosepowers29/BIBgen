"""
Histogram analysis for beam-induced background validation.
"""

from __future__ import annotations

import numpy as np
import h5py
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from BIBgen.analysis import plotting
from BIBgen.analysis.utils import eta_from_cylindrical, deltaR

class ComparisonAnalyzer:
    """Histogram analyzer for BIB detector hits in cylindrical coordinates."""
    
    def __init__(self,
        energy_range : tuple,
        phi_range : tuple,
        eta_range : tuple,
        s_range : tuple,
        z_range : tuple,
        time_range : Optional[tuple] = None,
        output_dir : str = "histograms"
    ):
        self.energy_range = energy_range
        self.phi_range = phi_range
        self.eta_range = eta_range
        self.s_range = s_range
        self.z_range = z_range
        self.time_range = time_range

        self.data = {}
        self.aggr_data = {}

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def aggregated_data(self, keys=None):
        """Per-dataset hits, with all events of a dataset concatenated together."""
        if keys is None:
            keys = set(self.data.keys())
        return {name : self.aggr_data[name] for name in self.data if name in keys}
    
    def load_from_training_data(self, filepath: str, event_ids: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Load data from training_data.hdf5 format.
        
        Expected structure:
            file.h5/training/evt_0  [n_hits, 4 or 5]  where features are [E, phi, s, z] or [E, phi, s, z, time]
            file.h5/validation/evt_0
            file.h5/test/evt_0

        Args:
            filepath: Path to HDF5 file
            event_ids: List of event IDs to load (None loads all)

        Returns:
            Dictionary with arrays: energy, phi, s, z, eta, and time (only if present in the data)
        """
        all_energy, all_phi, all_s, all_z, all_time = [], [], [], [], []

        with h5py.File(filepath, 'r') as f:
            if event_ids is None:
                events_to_load = []
                for group_name in ['training', 'validation', 'test']:
                    if group_name in f:
                        events_to_load.extend([f"{group_name}/{evt}" for evt in f[group_name].keys()])
            else:
                events_to_load = event_ids

            for event_path in events_to_load:
                try:
                    data = f[event_path][:]
                    all_energy.append(data[:, 0])
                    all_phi.append(data[:, 1])
                    all_s.append(data[:, 2])
                    all_z.append(data[:, 3])
                    if data.shape[1] >= 5:
                        all_time.append(data[:, 4])
                except KeyError:
                    continue

        energy = np.concatenate(all_energy)
        phi = np.concatenate(all_phi)
        s = np.concatenate(all_s)
        z = np.concatenate(all_z)
        eta = eta_from_cylindrical(s, z)

        result = {'energy': energy, 'phi': phi, 's': s, 'z': z, 'eta': eta}
        if all_time:
            result['time'] = np.concatenate(all_time)
        return result
    
    def load_from_raw_hdf5(self, filepath: str, event_ids: List[str] = None, 
                           collections: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Load data from raw SLCIO-converted HDF5.
        
        Args:
            filepath: Path to HDF5 file
            event_ids: List of event IDs
            collections: List of collection paths (defaults to ECal collections)
                        
        Returns:
            Dictionary with arrays: energy, phi, s, z, eta, and time (only if hit_times is present)
        """
        if collections is None:
            collections = [
                'ECalColls/ECalBarrelCollection',
                'ECalColls/ECalEndcapCollection'
            ]

        all_energy, all_x, all_y, all_z, all_time = [], [], [], [], []

        with h5py.File(filepath, 'r') as f:
            if event_ids is None:
                event_ids = [k for k in f.keys() if k.startswith('evt_')]

            for evt_id in event_ids:
                if evt_id not in f:
                    continue

                for coll_path in collections:
                    full_path = f"{evt_id}/{coll_path}"
                    if full_path not in f:
                        continue

                    coll = f[full_path]

                    if 'hit_energy' in coll:
                        energy = coll['hit_energy'][:]
                    elif 'hit_e_dep' in coll:
                        energy = coll['hit_e_dep'][:]
                    else:
                        continue

                    x = coll['hit_x_pos'][:]
                    y = coll['hit_y_pos'][:]
                    z = coll['hit_z_pos'][:]

                    all_energy.append(energy)
                    all_x.append(x)
                    all_y.append(y)
                    all_z.append(z)
                    if 'hit_times' in coll:
                        all_time.append(coll['hit_times'][:])

        energy = np.concatenate(all_energy)
        x = np.concatenate(all_x)
        y = np.concatenate(all_y)
        z = np.concatenate(all_z)

        phi = np.arctan2(y, x)
        s = np.sqrt(x**2 + y**2)
        eta = eta_from_cylindrical(s, z)

        result = {'energy': energy, 'phi': phi, 's': s, 'z': z, 'eta': eta}
        if all_time:
            result['time'] = np.concatenate(all_time)
        return result
    
    def load_from_dict(self, name : str, events : dict, 
                               is_sphered: bool = True,
                               sphering = None,
                               exponentiate_energy: bool = False) -> Dict[str, np.ndarray]:
        """
        Load data from model output arrays, one per event, and register it under `name`.
        
        Args:
            name: Key the dataset is registered under, also used as the plot legend label
            events: Mapping of event ID to array of shape (n_hits, 4 or 5) containing [E, phi, s, z] or [E, phi, s, z, time]
            is_sphered: Whether data is normalized
            sphering: Sphering object for unnormalization (required if is_sphered=True)
            exponentiate_energy: Whether the energy feature is ln(E) and must be exponentiated

        Returns:
            Dictionary with arrays: energy, phi, s, z, eta, and time (only if present in the data)
        """
        events_processed = {}
        aggr_processed = {'energy': [], 'phi': [], 's': [], 'z': [], 'eta': [], 'time': []}
        for event_id in events:
            if is_sphered:
                if sphering is None:
                    raise ValueError("Sphering object required for normalized data")
                data = sphering.untransform(events[event_id])
            else:
                data = events[event_id]

            energy = data[:, 0]
            if exponentiate_energy:
                energy = np.exp(energy)
            phi = data[:, 1]
            s = data[:, 2]
            z = data[:, 3]
            eta = eta_from_cylindrical(s, z)

            events_processed[event_id] = {'energy': energy, 'phi': phi, 's': s, 'z': z, 'eta': eta}
            if data.shape[1] >= 5:
                events_processed[event_id]['time'] = data[:, 4]
            for var in events_processed[event_id]:
                aggr_processed[var].append(events_processed[event_id][var])

        self.data[name] = events_processed
        self.aggr_data[name] = {var : np.concatenate(aggr_processed[var], axis=0) for var in aggr_processed if aggr_processed[var]}
        return self.aggr_data[name]

    def load_hits(self, name : str, hits : Dict[str, np.ndarray], event_id : str = "evt_0"):
        """
        Register an already-unpacked hits dictionary (energy, phi, s, z, eta) as a
        single-event dataset under `name`.
        """
        hits = {var : np.asarray(hits[var]) for var in hits}
        self.data[name] = {event_id : hits}
        self.aggr_data[name] = hits
        return hits
    
    def compute_hits_in_delta_r_cone(self, hits: Dict[str, np.ndarray],
                                     delta_r_threshold: float = 0.4,
                                     max_hits_sample: Optional[int] = 10000) -> np.ndarray:
        """
        Count neighboring hits within Delta R cone for each hit.
        
        Args:
            hits: Dictionary with eta and phi arrays
            delta_r_threshold: Delta R cone radius
            max_hits_sample: Maximum hits to analyze for computational efficiency
            
        Returns:
            Array of neighbor counts
        """
        eta = hits['eta']
        phi = hits['phi']
        n_hits = len(eta)
        
        if max_hits_sample is not None and n_hits > max_hits_sample:
            indices = np.random.choice(n_hits, max_hits_sample, replace=False)
            eta = eta[indices]
            phi = phi[indices]
            n_hits = max_hits_sample
        
        hits_in_cone = np.zeros(n_hits, dtype=int)
        
        for i in range(n_hits):
            delta_r = deltaR(eta[i], phi[i], eta, phi)
            hits_in_cone[i] = np.sum((delta_r < delta_r_threshold) & (delta_r > 0))
        
        return hits_in_cone
    
    def plot_basic_observables(self, hits: Dict[str, np.ndarray],
                               prefix: str = "bib",
                               bins: int = 100):
        """Generate histograms for basic observables."""
        has_time = 'time' in hits and self.time_range is not None
        ncols = 4 if has_time else 3
        fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 10))
        fig.suptitle('BIB Detector Hit Observables', fontsize=16, fontweight='bold')

        # Energy
        ax = axes[0, 0]
        ax.hist(hits['energy'], bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Energy [GeV]', fontsize=12)
        ax.set_ylabel('Hits', fontsize=12)
        ax.set_title('Energy Distribution', fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Phi
        ax = axes[0, 1]
        ax.hist(hits['phi'], bins=bins, edgecolor='black', alpha=0.7, color='coral')
        ax.set_xlabel('φ [rad]', fontsize=12)
        ax.set_ylabel('Hits', fontsize=12)
        ax.set_title('φ Distribution', fontsize=13, fontweight='bold')
        ax.set_xlim(-np.pi, np.pi)
        ax.grid(True, alpha=0.3)

        # Eta
        ax = axes[0, 2]
        ax.hist(hits['eta'], bins=bins, edgecolor='black', alpha=0.7, color='mediumseagreen')
        ax.set_xlabel('η', fontsize=12)
        ax.set_ylabel('Hits', fontsize=12)
        ax.set_title('η Distribution', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Time
        if has_time:
            ax = axes[0, 3]
            ax.hist(hits['time'], bins=bins, range=self.time_range, edgecolor='black', alpha=0.7, color='indianred')
            ax.set_xlabel('Time [ns]', fontsize=12)
            ax.set_ylabel('Hits', fontsize=12)
            ax.set_title('Time Distribution', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Radial
        ax = axes[1, 0]
        ax.hist(hits['s'], bins=bins, edgecolor='black', alpha=0.7, color='mediumpurple')
        ax.set_xlabel('Radial Distance [mm]', fontsize=12)
        ax.set_ylabel('Hits', fontsize=12)
        ax.set_title('Radial Distribution', fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Z position
        ax = axes[1, 1]
        ax.hist(hits['z'], bins=bins, edgecolor='black', alpha=0.7, color='gold')
        ax.set_xlabel('Z [mm]', fontsize=12)
        ax.set_ylabel('Hits', fontsize=12)
        ax.set_title('Z Distribution', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if has_time:
            axes[1, 2].axis('off')

        # Statistics
        ax = axes[1, 3] if has_time else axes[1, 2]
        ax.axis('off')
        stats_text = f"""
        Total Hits: {len(hits['energy']):,}
        
        Energy:
          Mean: {np.mean(hits['energy']):.3f} GeV
          Std: {np.std(hits['energy']):.3f} GeV
        
        Coverage:
          η: [{np.min(hits['eta']):.2f}, {np.max(hits['eta']):.2f}]
          s: [{np.min(hits['s']):.1f}, {np.max(hits['s']):.1f}] mm
          z: [{np.min(hits['z']):.1f}, {np.max(hits['z']):.1f}] mm
        """
        ax.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}_basic_observables.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_eta_phi_2d(self,
        name : str,
        prefix : str = "",
        bins: int | tuple = 50
    ):
        """Generate 2D histogram of eta vs phi."""
        outname = prefix + "_eta_phi_2d.png" if prefix != "" else "eta_phi_2d.png"
        outpath = self.output_dir / outname
        plotting.maia_hist2d(
            np.histogram2d(
                self.aggr_data[name]["eta"],
                self.aggr_data[name]["phi"],
                bins=bins,
                range=(self.eta_range, self.phi_range)
            ),
            outpath,
            r"$\eta$",
            r"$\phi$",
            self.eta_range,
            self.phi_range,
            mask_zero=True
        )

    def plot_s_eta_2d(self,
        name : str,
        prefix : str = "",
        bins: int | tuple = 50
    ):
        outname = prefix + "_s_eta_2d.png" if prefix != "" else "s_eta_2d.png"
        outpath = self.output_dir / outname
        plotting.maia_hist2d(
            np.histogram2d(
                self.aggr_data[name]["s"],
                self.aggr_data[name]["eta"],
                bins=bins,
                range=(self.s_range, self.eta_range)
            ),
            outpath,
            "s [mm]",
            r"$\eta$",
            self.s_range,
            self.eta_range,
            mask_zero=True
        )
        

    def plot_delta_r_clustering(self, hits: Dict[str, np.ndarray],
                                prefix: str = "bib",
                                delta_r_values: List[float] = [0.1, 0.2, 0.4, 0.6],
                                max_hits_sample: int = 10000,
                                bins: int = 50):
        """Generate Delta R clustering histograms."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Hit Clustering: Neighbors within ΔR Cone', 
                     fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for idx, delta_r in enumerate(delta_r_values):
            if idx >= 4:
                break
            
            hits_in_cone = self.compute_hits_in_delta_r_cone(
                hits, delta_r_threshold=delta_r, max_hits_sample=max_hits_sample
            )
            
            ax = axes[idx]
            ax.hist(hits_in_cone, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_xlabel('Neighbors within ΔR', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'ΔR < {delta_r}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            mean_val = np.mean(hits_in_cone)
            median_val = np.median(hits_in_cone)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
            ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}_delta_r_clustering.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_histograms(self, name: str,
                                prefix: str = "bib",
                                max_hits_for_clustering: int = 10000):
        """Generate all validation histograms for the dataset registered under `name`."""
        hits = self.aggr_data[name]
        self.plot_basic_observables(hits, prefix)
        self.plot_eta_phi_2d(name, prefix)
        self.plot_delta_r_clustering(hits, prefix, max_hits_sample=max_hits_for_clustering)
    
    def plot_kinematics_1d(self,
        name1 : str,
        name2 : str,
                                prefix: str = "comparison",
                                bins: int = 100,
                                normalized : bool = True,
                                log_scale : bool = True):
        """Plot MC and generated data overlayed on same axes."""
        mc_hits = self.aggr_data[name1]
        gen_hits = self.aggr_data[name2]

        has_time = 'time' in mc_hits and 'time' in gen_hits and self.time_range is not None
        ncols = 4 if has_time else 3
        fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 12))
        fig.suptitle('MC vs Generated', fontsize=16, fontweight='bold')

        # Energy
        axes[0, 0].hist(mc_hits['energy'], bins=bins, histtype='step', linewidth=2, 
                       label='MC', color='blue', alpha=0.7, density=normalized, range=self.energy_range)
        axes[0, 0].hist(gen_hits['energy'], bins=bins, histtype='step', linewidth=2, 
                       label='Generated', color='red', alpha=0.7, density=normalized, range=self.energy_range)
        axes[0, 0].set_xlabel('Energy [GeV]', fontsize=12)
        axes[0, 0].set_ylabel('Counts', fontsize=12)
        axes[0, 0].set_title('Energy', fontsize=13, fontweight='bold')
        axes[0, 0].set_xlim(*self.energy_range)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Phi
        axes[0, 1].hist(mc_hits['phi'], bins=bins, histtype='step', linewidth=2,
                       label='MC', color='blue', alpha=0.7, range=self.phi_range, density=normalized)
        axes[0, 1].hist(gen_hits['phi'], bins=bins, histtype='step', linewidth=2,
                       label='Generated', color='red', alpha=0.7, range=self.phi_range, density=normalized)
        axes[0, 1].set_xlabel('φ [rad]', fontsize=12)
        axes[0, 1].set_ylabel('Counts', fontsize=12)
        axes[0, 1].set_title('φ', fontsize=13, fontweight='bold')
        axes[0, 1].set_xlim(*self.phi_range)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Eta
        eta_mc = mc_hits['eta'][np.isfinite(mc_hits['eta'])]
        eta_gen = gen_hits['eta'][np.isfinite(gen_hits['eta'])]
        axes[0, 2].hist(eta_mc, bins=bins, histtype='step', linewidth=2,
                       label='MC', color='blue', alpha=0.7, density=normalized, range=self.eta_range)
        axes[0, 2].hist(eta_gen, bins=bins, histtype='step', linewidth=2,
                       label='Generated', color='red', alpha=0.7, density=normalized, range=self.eta_range)
        axes[0, 2].set_xlabel('η', fontsize=12)
        axes[0, 2].set_ylabel('Counts', fontsize=12)
        axes[0, 2].set_title('η', fontsize=13, fontweight='bold')
        axes[0, 2].set_xlim(*self.eta_range)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Time
        if has_time:
            axes[0, 3].hist(mc_hits['time'], bins=bins, histtype='step', linewidth=2,
                           label='MC', color='blue', alpha=0.7, density=normalized, range=self.time_range)
            axes[0, 3].hist(gen_hits['time'], bins=bins, histtype='step', linewidth=2,
                           label='Generated', color='red', alpha=0.7, density=normalized, range=self.time_range)
            axes[0, 3].set_xlabel('Time [ns]', fontsize=12)
            axes[0, 3].set_ylabel('Counts', fontsize=12)
            axes[0, 3].set_title('Time', fontsize=13, fontweight='bold')
            axes[0, 3].set_xlim(*self.time_range)
            axes[0, 3].legend()
            axes[0, 3].grid(True, alpha=0.3)

        # Radial
        axes[1, 0].hist(mc_hits['s'], bins=bins, histtype='step', linewidth=2,
                       label='MC', color='blue', alpha=0.7, density=normalized, range=self.s_range)
        axes[1, 0].hist(gen_hits['s'], bins=bins, histtype='step', linewidth=2,
                       label='Generated', color='red', alpha=0.7, density=normalized, range=self.s_range)
        axes[1, 0].set_xlabel('s [mm]', fontsize=12)
        axes[1, 0].set_ylabel('Counts', fontsize=12)
        axes[1, 0].set_title('Radial', fontsize=13, fontweight='bold')
        axes[1, 0].set_xlim(*self.s_range)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Z
        axes[1, 1].hist(mc_hits['z'], bins=bins, histtype='step', linewidth=2,
                       label='MC', color='blue', alpha=0.7, density=normalized, range=self.z_range)
        axes[1, 1].hist(gen_hits['z'], bins=bins, histtype='step', linewidth=2,
                       label='Generated', color='red', alpha=0.7, density=normalized, range=self.z_range)
        axes[1, 1].set_xlabel('z [mm]', fontsize=12)
        axes[1, 1].set_ylabel('Counts', fontsize=12)
        axes[1, 1].set_title('Z Position', fontsize=13, fontweight='bold')
        axes[1, 1].set_xlim(*self.z_range)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        if log_scale:
            axes[0, 0].set_yscale('log')
            axes[0, 1].set_yscale('log')
            axes[0, 2].set_yscale('log')
            axes[1, 0].set_yscale('log')
            axes[1, 1].set_yscale('log')

        if has_time:
            axes[1, 2].axis('off')

        # Stats
        stats_ax = axes[1, 3] if has_time else axes[1, 2]
        stats_ax.axis('off')
        stats = f"""
        MC vs Generated
        ───────────────
        
        Hits:
          MC:  {len(mc_hits['energy']):,}
          Gen: {len(gen_hits['energy']):,}
        
        Energy mean:
          MC:  {np.mean(mc_hits['energy']):.3f}
          Gen: {np.mean(gen_hits['energy']):.3f}
        
        η range:
          MC:  [{np.min(eta_mc):.2f}, {np.max(eta_mc):.2f}]
          Gen: [{np.min(eta_gen):.2f}, {np.max(eta_gen):.2f}]
        """
        stats_ax.text(0.1, 0.5, stats, fontsize=10, verticalalignment='center',
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}_overlay.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_kinematics_1d_with_residuals(self,
        name1 : str,
        name2 : str,
                                prefix: str = "comparison",
                                bins: int = 100,
                                normalized : bool = False,
                                log_scale : bool = True):
        """
        Same variables as plot_kinematics_1d (energy, phi, eta, s, z, and time
        when present in both datasets and self.time_range is set), but adds a
        residual panel (Gen - MC, per bin) beneath each histogram, sharing the
        same x-axis and bin edges as the panel above it.
        """
        import matplotlib.gridspec as gridspec

        mc_hits = self.aggr_data[name1]
        gen_hits = self.aggr_data[name2]

        has_time = 'time' in mc_hits and 'time' in gen_hits and self.time_range is not None
        ncols = 4 if has_time else 3
        fig = plt.figure(figsize=(6 * ncols, 13))
        gs = gridspec.GridSpec(4, ncols, height_ratios=[3, 1, 3, 1], hspace=0.5, wspace=0.3)
        fig.suptitle('MC vs Generated (with residuals)', fontsize=16, fontweight='bold')

        # (row of main panel, col, data key, xlabel, title, xlim, log-scale this panel)
        panel_specs = [
            (0, 0, 'energy', 'Energy [GeV]', 'Energy',     self.energy_range, True),
            (0, 1, 'phi',    'φ [rad]',      'φ',          self.phi_range,    False),
            (0, 2, 'eta',    'η',            'η',          self.eta_range,    False),
            (2, 0, 's',      's [mm]',       'Radial',     self.s_range,      True),
            (2, 1, 'z',      'z [mm]',       'Z Position', self.z_range,      False),
        ]
        if has_time:
            panel_specs.append((0, 3, 'time', 'Time [ns]', 'Time', self.time_range, False))

        for row0, col, key, xlabel, title, xlim, panel_log in panel_specs:
            ax_main = fig.add_subplot(gs[row0, col])
            ax_res = fig.add_subplot(gs[row0 + 1, col], sharex=ax_main)

            mc_vals = mc_hits[key]
            gen_vals = gen_hits[key]
            if key == 'eta':
                mc_vals = mc_vals[np.isfinite(mc_vals)]
                gen_vals = gen_vals[np.isfinite(gen_vals)]

            mc_counts, edges = np.histogram(mc_vals, bins=bins, range=xlim, density=normalized)
            gen_counts, _ = np.histogram(gen_vals, bins=edges, density=normalized)
            centers = 0.5 * (edges[:-1] + edges[1:])

            ax_main.step(centers, mc_counts, where='mid', color='blue', linewidth=2, label=name1)
            ax_main.step(centers, gen_counts, where='mid', color='red', linewidth=2, label=name2)
            ax_main.set_ylabel('Counts', fontsize=11)
            ax_main.set_title(title, fontsize=13, fontweight='bold')
            ax_main.set_xlim(*xlim)
            if log_scale and panel_log:
                ax_main.set_yscale('log')
            ax_main.legend(fontsize=9)
            ax_main.grid(True, alpha=0.3)
            plt.setp(ax_main.get_xticklabels(), visible=False)  # avoid duplicate x-labels above the residual panel

            residual = gen_counts - mc_counts
            ax_res.axhline(0, color='black', linewidth=1, linestyle='--')
            ax_res.bar(centers, residual, width=(edges[1] - edges[0]), color='gray', alpha=0.7)
            ax_res.set_xlabel(xlabel, fontsize=11)
            ax_res.set_ylabel('Gen - MC', fontsize=9)
            ax_res.set_xlim(*xlim)
            ax_res.grid(True, alpha=0.3)

        # Stats panel spans the s/z row block, third column -- no residual needed here
        ax_stats = fig.add_subplot(gs[2:4, 2])
        ax_stats.axis('off')
        eta_mc = mc_hits['eta'][np.isfinite(mc_hits['eta'])]
        eta_gen = gen_hits['eta'][np.isfinite(gen_hits['eta'])]
        stats = f"""
        MC vs Generated
        ───────────────

        Hits:
            MC:  {len(mc_hits['energy']):,}
            Gen: {len(gen_hits['energy']):,}

        Energy mean:
            MC:  {np.mean(mc_hits['energy']):.3f}
            Gen: {np.mean(gen_hits['energy']):.3f}

        η range:
            MC:  [{np.min(eta_mc):.2f}, {np.max(eta_mc):.2f}]
            Gen: [{np.min(eta_gen):.2f}, {np.max(eta_gen):.2f}]
        """
        ax_stats.text(0.1, 0.5, stats, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.savefig(self.output_dir / f"{prefix}_overlay_residuals.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_clustering(self,
        event_id : str,
        data_keys : set = None,
        reference_key : str = None,
        prefix : str = "",
        bins : int = 50,
        dR_range = (0.001, 0.2),
        use_energy : bool = True,
    ):
        # Use all keys available
        if data_keys is None:
            data_keys = set(self.data.keys())

        histograms = {}
        for name in data_keys:
            etas = self.data[name][event_id]["eta"]
            phis = self.data[name][event_id]["phi"]
            energies = self.data[name][event_id]["energy"]
            nhits = len(etas)
            event_hist = np.empty((nhits, bins))

            for ihit in range(nhits):
                dRs = deltaR(etas[ihit], phis[ihit], etas, phis)
                event_hist[ihit], bin_edges = np.histogram(dRs, bins=bins, range=dR_range, weights=energies) if use_energy else np.histogram(dRs, bins=bins, range=dR_range)

            histograms[name] = (np.mean(event_hist, axis=0), bin_edges)

        ylabel_base = "Average energy [GeV]" if use_energy else "Average # of hits"
        outname = prefix + "_clustering.png" if prefix != "" else "clustering.png"
        outpath = self.output_dir / outname
        plotting.maia_hist1d(histograms, outpath, r"$\Delta R$", ylabel_base)

        if reference_key is not None:
            ratio_histograms = {name : (histograms[name][0] / histograms[reference_key][0], histograms[name][1]) for name in histograms if name != reference_key}
            outname = prefix + "_clustering_ratio.png" if prefix != "" else "clustering_ratio.png"
            outpath = self.output_dir / outname
            plotting.maia_hist1d(ratio_histograms, outpath, r"$\Delta R$", "{} / {} ({})".format(ylabel_base, ylabel_base, reference_key), ybounds=(0.8, 1.2))

def compare_mc_vs_generated(mc_hits: Dict[str, np.ndarray],
                           gen_hits: Dict[str, np.ndarray],
                           output_dir: str = "comparison",
                           overlay: bool = True, **kwargs):
    """Compare MC vs generated. Set overlay=False for separate plots."""
    analyzer = ComparisonAnalyzer(output_dir=output_dir, **kwargs)
    analyzer.load_hits("MC", mc_hits)
    analyzer.load_hits("Generated", gen_hits)
    
    if overlay:
        analyzer.plot_kinematics_1d("MC", "Generated")
    else:
        analyzer.generate_all_histograms("MC", prefix="mc_truth")
        analyzer.generate_all_histograms("Generated", prefix="generated")
