"""
Histogram analysis for beam-induced background validation.
"""

from __future__ import annotations

import numpy as np
import h5py
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class BIBgenHistogramAnalyzer:
    """Histogram analyzer for BIB detector hits in cylindrical coordinates."""
    
    def __init__(self, output_dir: str = "histograms"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_eta_from_cylindrical(self, s: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute pseudorapidity from cylindrical coordinates.
        
        Args:
            s: Radial distance from beam axis
            z: Z-position along beam axis
            
        Returns:
            Pseudorapidity eta
        """
        theta = abs(np.arctan2(s, z))
        eta = -np.log(np.tan((theta % (2*np.pi)) / 2.0 + 1e-10))
        return eta
    
    def load_from_training_data(self, filepath: str, event_ids: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Load data from training_data.hdf5 format.
        
        Expected structure:
            file.h5/training/evt_0  [n_hits, 4]  where features are [E, phi, s, z]
            file.h5/validation/evt_0
            file.h5/test/evt_0
        
        Args:
            filepath: Path to HDF5 file
            event_ids: List of event IDs to load (None loads all)
                      
        Returns:
            Dictionary with arrays: energy, phi, s, z, eta
        """
        all_energy, all_phi, all_s, all_z = [], [], [], []
        
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
                except KeyError:
                    continue
        
        energy = np.concatenate(all_energy)
        phi = np.concatenate(all_phi)
        s = np.concatenate(all_s)
        z = np.concatenate(all_z)
        eta = self.compute_eta_from_cylindrical(s, z)
        
        return {'energy': energy, 'phi': phi, 's': s, 'z': z, 'eta': eta}
    
    def load_from_raw_hdf5(self, filepath: str, event_ids: List[str] = None, 
                           collections: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Load data from raw SLCIO-converted HDF5.
        
        Args:
            filepath: Path to HDF5 file
            event_ids: List of event IDs
            collections: List of collection paths (defaults to ECal collections)
                        
        Returns:
            Dictionary with arrays: energy, phi, s, z, eta
        """
        if collections is None:
            collections = [
                'ECalColls/ECalBarrelCollection',
                'ECalColls/ECalEndcapCollection'
            ]
        
        all_energy, all_x, all_y, all_z = [], [], [], []
        
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
        
        energy = np.concatenate(all_energy)
        x = np.concatenate(all_x)
        y = np.concatenate(all_y)
        z = np.concatenate(all_z)
        
        phi = np.arctan2(y, x)
        s = np.sqrt(x**2 + y**2)
        eta = self.compute_eta_from_cylindrical(s, z)
        
        return {'energy': energy, 'phi': phi, 's': s, 'z': z, 'eta': eta}
    
    def load_from_model_output(self, model_output: np.ndarray, 
                               is_sphered: bool = True,
                               sphering = None) -> Dict[str, np.ndarray]:
        """
        Load data from model output array.
        
        Args:
            model_output: Array with shape (n_hits, 4) containing [E, phi, s, z]
            is_sphered: Whether data is normalized
            sphering: Sphering object for unnormalization (required if is_sphered=True)
                     
        Returns:
            Dictionary with arrays: energy, phi, s, z, eta
        """
        if is_sphered:
            if sphering is None:
                raise ValueError("Sphering object required for normalized data")
            data = sphering.untransform(model_output)
        else:
            data = model_output
        
        energy = data[:, 0]
        phi = data[:, 1]
        s = data[:, 2]
        z = data[:, 3]
        eta = self.compute_eta_from_cylindrical(s, z)
        
        return {'energy': energy, 'phi': phi, 's': s, 'z': z, 'eta': eta}
    
    def compute_delta_r(self, eta1: np.ndarray, phi1: np.ndarray,
                       eta2: np.ndarray, phi2: np.ndarray) -> np.ndarray:
        """Compute Delta R metric between coordinate pairs."""
        delta_eta = eta1 - eta2
        delta_phi = phi1 - phi2
        delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi))
        return np.sqrt(delta_eta**2 + delta_phi**2)
    
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
            delta_r = self.compute_delta_r(eta[i], phi[i], eta, phi)
            hits_in_cone[i] = np.sum((delta_r < delta_r_threshold) & (delta_r > 0))
        
        return hits_in_cone
    
    def plot_basic_observables(self, hits: Dict[str, np.ndarray],
                               prefix: str = "bib",
                               bins: int = 100):
        """Generate histograms for basic observables."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
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
        
        # Statistics
        ax = axes[1, 2]
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
    
    def plot_eta_phi_2d(self, hits: Dict[str, np.ndarray],
                        prefix: str = "bib",
                        bins: int = 100):
        """Generate 2D histogram of eta vs phi."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        h = ax.hist2d(hits['eta'], hits['phi'], bins=bins, cmap='viridis', cmin=1)
        plt.colorbar(h[3], ax=ax, label='Hits')
        
        ax.set_xlabel('η', fontsize=12)
        ax.set_ylabel('φ [rad]', fontsize=12)
        ax.set_title('Hit Distribution in η-φ Space', fontsize=14, fontweight='bold')
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xlim(-3, 3)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}_eta_phi_2d.png", dpi=300, bbox_inches='tight')
        plt.close()
    
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
    
    def generate_all_histograms(self, hits: Dict[str, np.ndarray],
                                prefix: str = "bib",
                                max_hits_for_clustering: int = 10000):
        """Generate all validation histograms."""
        self.plot_basic_observables(hits, prefix)
        self.plot_eta_phi_2d(hits, prefix)
        self.plot_delta_r_clustering(hits, prefix, max_hits_sample=max_hits_for_clustering)
    
    def plot_overlay_comparison(self, mc_hits: Dict[str, np.ndarray],
                                gen_hits: Dict[str, np.ndarray],
                                prefix: str = "comparison",
                                bins: int = 100,
                                normalized : bool = True):
        """Plot MC and generated data overlayed on same axes."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MC vs Generated', fontsize=16, fontweight='bold')
        
        # Energy
        axes[0, 0].hist(mc_hits['energy'], bins=bins, histtype='step', linewidth=2, 
                       label='MC', color='blue', alpha=0.7, density=normalized)
        axes[0, 0].hist(gen_hits['energy'], bins=bins, histtype='step', linewidth=2, 
                       label='Generated', color='red', alpha=0.7, density=normalized)
        axes[0, 0].set_xlabel('Energy [GeV]', fontsize=12)
        axes[0, 0].set_ylabel('Counts', fontsize=12)
        axes[0, 0].set_title('Energy', fontsize=13, fontweight='bold')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_xlim(0, 0.04)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Phi
        axes[0, 1].hist(mc_hits['phi'], bins=bins, histtype='step', linewidth=2,
                       label='MC', color='blue', alpha=0.7, range=(-np.pi, np.pi), density=normalized)
        axes[0, 1].hist(gen_hits['phi'], bins=bins, histtype='step', linewidth=2,
                       label='Generated', color='red', alpha=0.7, range=(-np.pi, np.pi), density=normalized)
        axes[0, 1].set_xlabel('φ [rad]', fontsize=12)
        axes[0, 1].set_ylabel('Counts', fontsize=12)
        axes[0, 1].set_title('φ', fontsize=13, fontweight='bold')
        axes[0, 1].set_xlim(-np.pi, np.pi)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Eta
        eta_mc = mc_hits['eta'][np.isfinite(mc_hits['eta'])]
        eta_gen = gen_hits['eta'][np.isfinite(gen_hits['eta'])]
        axes[0, 2].hist(eta_mc, bins=bins, histtype='step', linewidth=2,
                       label='MC', color='blue', alpha=0.7, density=normalized)
        axes[0, 2].hist(eta_gen, bins=bins, histtype='step', linewidth=2,
                       label='Generated', color='red', alpha=0.7, density=normalized)
        axes[0, 2].set_xlabel('η', fontsize=12)
        axes[0, 2].set_ylabel('Counts', fontsize=12)
        axes[0, 2].set_title('η', fontsize=13, fontweight='bold')
        axes[0, 2].set_xlim(-3, 3)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Radial
        axes[1, 0].hist(mc_hits['s'], bins=bins, histtype='step', linewidth=2,
                       label='MC', color='blue', alpha=0.7, density=normalized)
        axes[1, 0].hist(gen_hits['s'], bins=bins, histtype='step', linewidth=2,
                       label='Generated', color='red', alpha=0.7, density=normalized)
        axes[1, 0].set_xlabel('s [mm]', fontsize=12)
        axes[1, 0].set_ylabel('Counts', fontsize=12)
        axes[1, 0].set_title('Radial', fontsize=13, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_xlim(0, 2500)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Z
        axes[1, 1].hist(mc_hits['z'], bins=bins, histtype='step', linewidth=2,
                       label='MC', color='blue', alpha=0.7, density=normalized)
        axes[1, 1].hist(gen_hits['z'], bins=bins, histtype='step', linewidth=2,
                       label='Generated', color='red', alpha=0.7, density=normalized)
        axes[1, 1].set_xlabel('z [mm]', fontsize=12)
        axes[1, 1].set_ylabel('Counts', fontsize=12)
        axes[1, 1].set_title('Z Position', fontsize=13, fontweight='bold')
        axes[1, 1].set_xlim(-2800, 2800)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Stats
        axes[1, 2].axis('off')
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
        axes[1, 2].text(0.1, 0.5, stats, fontsize=10, verticalalignment='center',
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}_overlay.png", dpi=300, bbox_inches='tight')
        plt.close()


def compare_mc_vs_generated(mc_hits: Dict[str, np.ndarray],
                           gen_hits: Dict[str, np.ndarray],
                           output_dir: str = "comparison",
                           overlay: bool = True):
    """Compare MC vs generated. Set overlay=False for separate plots."""
    analyzer = BIBgenHistogramAnalyzer(output_dir=output_dir)
    
    if overlay:
        analyzer.plot_overlay_comparison(mc_hits, gen_hits)
    else:
        analyzer.generate_all_histograms(mc_hits, prefix="mc_truth")
        analyzer.generate_all_histograms(gen_hits, prefix="generated")
