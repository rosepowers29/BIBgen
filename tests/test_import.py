"""Basic tests for BIBgen package."""
import pytest

def test_package_import():
    """Test that the BIBgen package can be imported."""
    try:
        import BIBgen
        assert True
    except ImportError:
        pytest.fail("BIBgen package could not be imported")


def test_package_has_name():
    """Test that the package has a __name__ attribute."""
    import BIBgen
    assert hasattr(BIBgen, '__name__')
    assert BIBgen.__name__ == 'BIBgen'
