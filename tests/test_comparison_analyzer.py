"""
Tests for the comparison analysis module.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

from BIBgen.analysis import ComparisonAnalyzer, compare_mc_vs_generated

var_range = {
    "energy_range" :(-0.0005, 0.005),
    "phi_range" : (-1.0, 1.0),
    "eta_range" : (-1.3, 1.3),
    "s_range" : (1800, 2250),
    "z_range" : (-2800, 2800),
}

var_range_with_time = {**var_range, "time_range": (0, 15000)}

def make_hits(n_hits, rng):
    """Sample hits that land inside var_range, so the 2D histograms are not empty."""
    s = rng.uniform(1850, 2200, n_hits)
    z = rng.uniform(-2500, 2500, n_hits)
    theta = np.abs(np.arctan2(s, z))
    return {
        'energy': rng.exponential(0.001, n_hits),
        'phi': rng.uniform(-1.0, 1.0, n_hits),
        's': s,
        'z': z,
        'eta': -np.log(np.tan(theta / 2.0)),
    }

def make_events(n_events, n_hits, rng):
    """Model-output style events: arrays of shape (n_hits, 4) holding [E, phi, s, z]."""
    return {
        f"evt_{i}": np.column_stack([
            rng.exponential(0.001, n_hits),
            rng.uniform(-1.0, 1.0, n_hits),
            rng.uniform(1850, 2200, n_hits),
            rng.uniform(-2500, 2500, n_hits),
        ])
        for i in range(n_events)
    }

def make_events_with_time(n_events, n_hits, rng):
    """Model-output style events: arrays of shape (n_hits, 5) holding [E, phi, s, z, time]."""
    return {
        f"evt_{i}": np.column_stack([
            rng.exponential(0.001, n_hits),
            rng.uniform(-1.0, 1.0, n_hits),
            rng.uniform(1850, 2200, n_hits),
            rng.uniform(-2500, 2500, n_hits),
            rng.uniform(0, 15000, n_hits),
        ])
        for i in range(n_events)
    }


class TestComparisonAnalyzer:
    """Test suite for the comparison analyzer."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(0)

    @pytest.fixture
    def sample_hits(self, rng):
        """Generate sample hit data for testing."""
        return make_hits(1000, rng)

    @pytest.fixture
    def loaded_analyzer(self, temp_dir, rng):
        """Analyzer with an 'MC' and a 'Generated' dataset already registered."""
        analyzer = ComparisonAnalyzer(**var_range, output_dir=temp_dir)
        analyzer.load_from_dict("MC", make_events(3, 50, rng), is_sphered=False)
        analyzer.load_from_dict("Generated", make_events(3, 50, rng), is_sphered=False)
        return analyzer

    def test_analyzer_creation(self, temp_dir):
        """Test that analyzer creates output directory."""
        analyzer = ComparisonAnalyzer(**var_range, output_dir=temp_dir)
        assert Path(temp_dir).exists()
        assert analyzer.output_dir == Path(temp_dir)

    def test_delta_r_cone_counting(self, sample_hits):
        """Test neighbor counting in delta R cone."""
        analyzer = ComparisonAnalyzer(**var_range)

        # Use small sample to speed up test
        small_hits = {k: v[:100] for k, v in sample_hits.items()}

        counts = analyzer.compute_hits_in_delta_r_cone(
            small_hits,
            delta_r_threshold=0.5,
            max_hits_sample=100
        )

        assert len(counts) == 100
        assert np.all(counts >= 0)
        assert counts.dtype == np.int_

    def test_plot_basic_observables(self, sample_hits, temp_dir):
        """Test basic observable plotting."""
        analyzer = ComparisonAnalyzer(**var_range, output_dir=temp_dir)
        analyzer.plot_basic_observables(sample_hits, prefix="test")

        output_file = Path(temp_dir) / "test_basic_observables.png"
        assert output_file.exists()

    def test_plot_eta_phi_2d(self, loaded_analyzer, temp_dir):
        """Test 2D eta-phi plotting."""
        loaded_analyzer.plot_eta_phi_2d("MC", prefix="test")

        output_file = Path(temp_dir) / "test_eta_phi_2d.png"
        assert output_file.exists()

    def test_plot_eta_phi_2d_default_name(self, loaded_analyzer, temp_dir):
        """An empty prefix drops the leading underscore from the filename."""
        loaded_analyzer.plot_eta_phi_2d("MC")

        assert (Path(temp_dir) / "eta_phi_2d.png").exists()

    def test_plot_s_eta_2d(self, loaded_analyzer, temp_dir):
        """Test 2D s-eta plotting."""
        loaded_analyzer.plot_s_eta_2d("MC", prefix="test")

        output_file = Path(temp_dir) / "test_s_eta_2d.png"
        assert output_file.exists()

    def test_plot_delta_r_clustering(self, sample_hits, temp_dir):
        """Test delta R clustering plots."""
        analyzer = ComparisonAnalyzer(**var_range, output_dir=temp_dir)

        # Use small sample for speed
        small_hits = {k: v[:200] for k, v in sample_hits.items()}

        analyzer.plot_delta_r_clustering(
            small_hits,
            prefix="test",
            max_hits_sample=200
        )

        output_file = Path(temp_dir) / "test_delta_r_clustering.png"
        assert output_file.exists()

    def test_generate_all_histograms(self, loaded_analyzer, temp_dir):
        """Test that all histograms are generated."""
        loaded_analyzer.generate_all_histograms("MC", prefix="test", max_hits_for_clustering=150)

        # Check all three expected files
        assert (Path(temp_dir) / "test_basic_observables.png").exists()
        assert (Path(temp_dir) / "test_eta_phi_2d.png").exists()
        assert (Path(temp_dir) / "test_delta_r_clustering.png").exists()

    def test_kinematics_1d(self, loaded_analyzer, temp_dir):
        """Test overlay comparison plotting."""
        loaded_analyzer.plot_kinematics_1d("MC", "Generated", prefix="test")

        output_file = Path(temp_dir) / "test_overlay.png"
        assert output_file.exists()

    def test_kinematics_1d_with_residuals(self, loaded_analyzer, temp_dir):
        """Test overlay comparison plotting with residual panels."""
        loaded_analyzer.plot_kinematics_1d_with_residuals("MC", "Generated", prefix="test")

        output_file = Path(temp_dir) / "test_overlay_residuals.png"
        assert output_file.exists()

    def test_plot_clustering(self, loaded_analyzer, temp_dir):
        """Test the per-event Delta R clustering profile, with the ratio to a reference."""
        loaded_analyzer.plot_clustering("evt_0", reference_key="MC", prefix="test", bins=10)

        assert (Path(temp_dir) / "test_clustering.png").exists()
        assert (Path(temp_dir) / "test_clustering_ratio.png").exists()

    def test_compute_wasserstein_nd_identical_is_near_zero(self, loaded_analyzer):
        """Comparing a dataset against itself yields ~0 distance for a 2D pair (no subsampling, so the point sets match exactly)."""
        d = loaded_analyzer.compute_wasserstein_nd("MC", "MC", ("eta", "phi"), max_samples=None)

        assert d == pytest.approx(0.0, abs=1e-8)

    def test_compute_wasserstein_nd_detects_shift(self, rng):
        """A shifted eta-phi distribution gives a positive distance roughly matching the shift."""
        analyzer = ComparisonAnalyzer(**var_range)
        analyzer.load_hits("MC", make_hits(100, rng))

        shifted = make_hits(100, rng)
        shifted["phi"] = shifted["phi"] + 0.5
        analyzer.load_hits("Shifted", shifted)

        d = analyzer.compute_wasserstein_nd("MC", "Shifted", ("eta", "phi"), max_samples=50, rng=np.random.default_rng(0))

        assert d > 0.1

    def test_compute_wasserstein_nd_pairs(self, loaded_analyzer):
        """compute_wasserstein_nd_pairs returns one distance per requested variable grouping."""
        pairs = [("eta", "phi"), ("s", "z")]
        result = loaded_analyzer.compute_wasserstein_nd_pairs("MC", "Generated", pairs, max_samples=30, seed=0)

        assert set(result.keys()) == set(pairs)
        assert all(v >= 0 for v in result.values())

    def test_compute_wasserstein_nd_ignores_non_finite_rows(self, temp_dir):
        """Non-finite rows (e.g. NaN eta) are dropped rather than propagating to the LP."""
        analyzer = ComparisonAnalyzer(**var_range, output_dir=temp_dir)
        hits = {
            'energy': np.array([1.0, 2.0, 3.0, 4.0]),
            'phi': np.array([0.0, 0.1, 0.2, 0.3]),
            's': np.array([1900.0, 1950.0, 2000.0, 2050.0]),
            'z': np.array([100.0, 150.0, 200.0, 250.0]),
            'eta': np.array([1.0, np.nan, 2.0, np.inf]),
        }
        analyzer.load_hits("A", hits)
        analyzer.load_hits("B", hits)

        d = analyzer.compute_wasserstein_nd("A", "B", ("eta", "phi"), max_samples=None)

        assert d == pytest.approx(0.0, abs=1e-8)

    def test_plot_clustering_without_energy_weights(self, loaded_analyzer, temp_dir):
        """use_energy=False falls back to plain hit counting."""
        loaded_analyzer.plot_clustering("evt_0", prefix="test", bins=10, use_energy=False)

        assert (Path(temp_dir) / "test_clustering.png").exists()

    def test_load_from_dict_unsphered(self, rng):
        """Test loading unsphered model output."""
        analyzer = ComparisonAnalyzer(**var_range)

        n_hits = 100
        events = make_events(2, n_hits, rng)

        hits = analyzer.load_from_dict("MC", events, is_sphered=False)

        assert 'energy' in hits
        assert 'phi' in hits
        assert 's' in hits
        assert 'z' in hits
        assert 'eta' in hits
        assert len(hits['energy']) == 2 * n_hits

        # Registered under the given name, both per-event and aggregated
        assert set(analyzer.data["MC"].keys()) == set(events.keys())
        assert analyzer.aggregated_data()["MC"] is analyzer.aggr_data["MC"]

    def test_load_from_dict_includes_time_when_present(self, rng):
        """A 5th (time) column in the events gets picked up when time_range is set."""
        analyzer = ComparisonAnalyzer(**var_range_with_time)

        n_hits = 50
        events = make_events_with_time(2, n_hits, rng)
        hits = analyzer.load_from_dict("MC", events, is_sphered=False)

        assert 'time' in hits
        assert len(hits['time']) == 2 * n_hits

    def test_load_from_dict_omits_time_when_absent(self, rng):
        """4-column (no time) events still load fine, with no 'time' key."""
        analyzer = ComparisonAnalyzer(**var_range)

        hits = analyzer.load_from_dict("MC", make_events(2, 50, rng), is_sphered=False)

        assert 'time' not in hits

    def test_plot_kinematics_1d_with_time(self, temp_dir, rng):
        """plot_kinematics_1d renders a time panel when both datasets carry it."""
        analyzer = ComparisonAnalyzer(**var_range_with_time, output_dir=temp_dir)
        analyzer.load_from_dict("MC", make_events_with_time(2, 50, rng), is_sphered=False)
        analyzer.load_from_dict("Generated", make_events_with_time(2, 50, rng), is_sphered=False)

        analyzer.plot_kinematics_1d("MC", "Generated", prefix="test_time")

        assert (Path(temp_dir) / "test_time_overlay.png").exists()

    def test_load_from_dict_requires_sphering(self, rng):
        """Sphered data without a sphering object is an error."""
        analyzer = ComparisonAnalyzer(**var_range)

        with pytest.raises(ValueError):
            analyzer.load_from_dict("MC", make_events(1, 10, rng), is_sphered=True)

    def test_load_from_dict_exponentiate_energy(self, rng):
        """exponentiate_energy undoes a ln(E) energy feature."""
        analyzer = ComparisonAnalyzer(**var_range)

        events = make_events(1, 20, rng)
        log_events = {k: v.copy() for k, v in events.items()}
        for v in log_events.values():
            v[:, 0] = np.log(v[:, 0])

        raw = analyzer.load_from_dict("raw", events, is_sphered=False)
        exp = analyzer.load_from_dict("log", log_events, is_sphered=False, exponentiate_energy=True)

        assert np.allclose(raw['energy'], exp['energy'])

    def test_compare_mc_vs_generated_overlay(self, sample_hits, temp_dir):
        """Test comparison function with overlay."""
        gen_hits = sample_hits.copy()
        gen_hits['energy'] = gen_hits['energy'] * 0.9

        compare_mc_vs_generated(sample_hits, gen_hits, output_dir=temp_dir, overlay=True, **var_range)

        assert (Path(temp_dir) / "comparison_overlay.png").exists()

    def test_compare_mc_vs_generated_separate(self, sample_hits, temp_dir):
        """Test comparison function with separate plots."""
        gen_hits = sample_hits.copy()

        compare_mc_vs_generated(sample_hits, gen_hits, output_dir=temp_dir, overlay=False, **var_range)

        assert (Path(temp_dir) / "mc_truth_basic_observables.png").exists()
        assert (Path(temp_dir) / "generated_basic_observables.png").exists()

    def test_handles_nan_in_eta(self, temp_dir):
        """Test that code handles NaN values in eta."""
        analyzer = ComparisonAnalyzer(**var_range, output_dir=temp_dir)

        hits = {
            'energy': np.array([1.0, 2.0, 3.0]),
            'phi': np.array([0.0, 1.0, 2.0]),
            's': np.array([100.0, 200.0, 300.0]),
            'z': np.array([100.0, 0.0, 100.0]),
            'eta': np.array([1.0, np.nan, 2.0])
        }

        # Should not crash
        analyzer.plot_basic_observables(hits, prefix="test")
        assert (Path(temp_dir) / "test_basic_observables.png").exists()

    def test_kinematics_1d_handles_nan_in_eta(self, temp_dir, rng):
        """NaN eta values are dropped before the overlay is drawn."""
        analyzer = ComparisonAnalyzer(**var_range, output_dir=temp_dir)

        mc_hits = make_hits(100, rng)
        gen_hits = make_hits(100, rng)
        gen_hits['eta'][::10] = np.nan

        analyzer.load_hits("MC", mc_hits)
        analyzer.load_hits("Generated", gen_hits)
        analyzer.plot_kinematics_1d("MC", "Generated", prefix="test")

        assert (Path(temp_dir) / "test_overlay.png").exists()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_hits(self):
        """Test behavior with empty hit arrays."""
        analyzer = ComparisonAnalyzer(**var_range)

        empty_hits = {
            'energy': np.array([]),
            'phi': np.array([]),
            's': np.array([]),
            'z': np.array([]),
            'eta': np.array([])
        }

        # Should return empty result without crashing
        counts = analyzer.compute_hits_in_delta_r_cone(empty_hits)
        assert len(counts) == 0

    def test_single_hit(self):
        """Test with single hit."""
        analyzer = ComparisonAnalyzer(**var_range)

        single_hit = {
            'energy': np.array([1.0]),
            'phi': np.array([0.0]),
            's': np.array([100.0]),
            'z': np.array([100.0]),
            'eta': np.array([0.8])
        }

        counts = analyzer.compute_hits_in_delta_r_cone(single_hit)
        assert len(counts) == 1
        assert counts[0] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
