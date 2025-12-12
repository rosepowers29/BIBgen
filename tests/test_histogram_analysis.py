"""
Tests for histogram analysis module.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

from BIBgen.analysis import BIBgenHistogramAnalyzer, compare_mc_vs_generated


class TestBIBgenHistogramAnalyzer:
    """Test suite for histogram analyzer."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)
    
    @pytest.fixture
    def sample_hits(self):
        """Generate sample hit data for testing."""
        n_hits = 1000
        return {
            'energy': np.random.exponential(0.5, n_hits),
            'phi': np.random.uniform(-np.pi, np.pi, n_hits),
            's': np.random.exponential(100, n_hits) + 50,
            'z': np.random.normal(0, 200, n_hits),
            'eta': np.random.normal(0, 2, n_hits)
        }
    
    def test_analyzer_creation(self, temp_dir):
        """Test that analyzer creates output directory."""
        analyzer = BIBgenHistogramAnalyzer(output_dir=temp_dir)
        assert Path(temp_dir).exists()
        assert analyzer.output_dir == Path(temp_dir)
    
    def test_eta_computation(self):
        """Test pseudorapidity calculation."""
        analyzer = BIBgenHistogramAnalyzer()
        
        # Test forward region (positive z)
        s = np.array([100.0])
        z = np.array([200.0])
        eta = analyzer.compute_eta_from_cylindrical(s, z)
        assert np.isfinite(eta[0])
        assert eta[0] > 0
        
        # Test backward region (negative z)
        z = np.array([-200.0])
        eta = analyzer.compute_eta_from_cylindrical(s, z)
        assert np.isfinite(eta[0])
        assert eta[0] < 0
    
    def test_delta_r_computation(self):
        """Test delta R calculation."""
        analyzer = BIBgenHistogramAnalyzer()
        
        eta1 = np.array([0.0, 1.0])
        phi1 = np.array([0.0, 0.0])
        eta2 = np.array([1.0, 1.0])
        phi2 = np.array([0.0, np.pi])
        
        dr = analyzer.compute_delta_r(eta1, phi1, eta2, phi2)
        
        # Check that delta R is positive
        assert np.all(dr >= 0)
        
        # Check that identical points have dr = 0
        dr_same = analyzer.compute_delta_r(eta1, phi1, eta1, phi1)
        assert np.allclose(dr_same, 0)
    
    def test_delta_r_cone_counting(self, sample_hits):
        """Test neighbor counting in delta R cone."""
        analyzer = BIBgenHistogramAnalyzer()
        
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
        analyzer = BIBgenHistogramAnalyzer(output_dir=temp_dir)
        analyzer.plot_basic_observables(sample_hits, prefix="test")
        
        output_file = Path(temp_dir) / "test_basic_observables.png"
        assert output_file.exists()
    
    def test_plot_eta_phi_2d(self, sample_hits, temp_dir):
        """Test 2D eta-phi plotting."""
        analyzer = BIBgenHistogramAnalyzer(output_dir=temp_dir)
        analyzer.plot_eta_phi_2d(sample_hits, prefix="test")
        
        output_file = Path(temp_dir) / "test_eta_phi_2d.png"
        assert output_file.exists()
    
    def test_plot_delta_r_clustering(self, sample_hits, temp_dir):
        """Test delta R clustering plots."""
        analyzer = BIBgenHistogramAnalyzer(output_dir=temp_dir)
        
        # Use small sample for speed
        small_hits = {k: v[:200] for k, v in sample_hits.items()}
        
        analyzer.plot_delta_r_clustering(
            small_hits, 
            prefix="test",
            max_hits_sample=200
        )
        
        output_file = Path(temp_dir) / "test_delta_r_clustering.png"
        assert output_file.exists()
    
    def test_generate_all_histograms(self, sample_hits, temp_dir):
        """Test that all histograms are generated."""
        analyzer = BIBgenHistogramAnalyzer(output_dir=temp_dir)
        
        small_hits = {k: v[:200] for k, v in sample_hits.items()}
        analyzer.generate_all_histograms(small_hits, prefix="test")
        
        # Check all three expected files
        assert (Path(temp_dir) / "test_basic_observables.png").exists()
        assert (Path(temp_dir) / "test_eta_phi_2d.png").exists()
        assert (Path(temp_dir) / "test_delta_r_clustering.png").exists()
    
    def test_overlay_comparison(self, sample_hits, temp_dir):
        """Test overlay comparison plotting."""
        analyzer = BIBgenHistogramAnalyzer(output_dir=temp_dir)
        
        # Create slightly different generated data
        gen_hits = sample_hits.copy()
        gen_hits['energy'] = gen_hits['energy'] * 1.1
        
        analyzer.plot_overlay_comparison(sample_hits, gen_hits, prefix="test")
        
        output_file = Path(temp_dir) / "test_overlay.png"
        assert output_file.exists()
    
    def test_load_from_model_output_unsphered(self):
        """Test loading unsphered model output."""
        analyzer = BIBgenHistogramAnalyzer()
        
        # Create fake model output
        n_hits = 100
        model_output = np.column_stack([
            np.random.exponential(0.5, n_hits),
            np.random.uniform(-np.pi, np.pi, n_hits),
            np.random.exponential(100, n_hits),
            np.random.normal(0, 200, n_hits)
        ])
        
        hits = analyzer.load_from_model_output(model_output, is_sphered=False)
        
        assert 'energy' in hits
        assert 'phi' in hits
        assert 's' in hits
        assert 'z' in hits
        assert 'eta' in hits
        assert len(hits['energy']) == n_hits
    
    def test_compare_mc_vs_generated_overlay(self, sample_hits, temp_dir):
        """Test comparison function with overlay."""
        gen_hits = sample_hits.copy()
        gen_hits['energy'] = gen_hits['energy'] * 0.9
        
        compare_mc_vs_generated(sample_hits, gen_hits, output_dir=temp_dir, overlay=True)
        
        assert (Path(temp_dir) / "comparison_overlay.png").exists()
    
    def test_compare_mc_vs_generated_separate(self, sample_hits, temp_dir):
        """Test comparison function with separate plots."""
        gen_hits = sample_hits.copy()
        
        compare_mc_vs_generated(sample_hits, gen_hits, output_dir=temp_dir, overlay=False)
        
        assert (Path(temp_dir) / "mc_truth_basic_observables.png").exists()
        assert (Path(temp_dir) / "generated_basic_observables.png").exists()
    
    def test_handles_nan_in_eta(self, temp_dir):
        """Test that code handles NaN values in eta."""
        analyzer = BIBgenHistogramAnalyzer(output_dir=temp_dir)
        
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


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_hits(self):
        """Test behavior with empty hit arrays."""
        analyzer = BIBgenHistogramAnalyzer()
        
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
        analyzer = BIBgenHistogramAnalyzer()
        
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
