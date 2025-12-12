import pytest
import h5py
import os
import sys
import BIBgen.slcio_to_hdf5 as converter


@pytest.fixture
def output_hdf5_file(tmp_path):
    """Defines the path for the output hdf5 file."""
    return tmp_path/"output.hdf5"

def inputfiles(files):
    filepaths = []
    for file in files:
        filepaths.append("/scratch/rosep8/BIBgen/tests/input_files/"+file)
    return filepaths

def test_create_hdf_file_single_input(output_hdf5_file):
    """Test that the converter creates the output HDF5 file for one file input."""
    infile = inputfiles(["BIB_sim_100.slcio"])
    converter.convert_files(infile, output_hdf5_file)
    assert os.path.exists(output_hdf5_file)

def test_create_hdf_file_multi_input(output_hdf5_file):
    """
    Test that the converter creates the output HDF5 file for multiple inputs.
    """
    infiles = inputfiles(["BIB_sim_100.slcio", "BIB_sim_200.slcio"]) 
    converter.convert_files(infiles, output_hdf5_file)
    assert os.path.exists(output_hdf5_file)

def test_hdf5_structure(output_hdf5_file):
    """
    Verify the internal structure of the HDF5 file (events, collections, hit variables)
    """
    inputfile = inputfiles(["BIB_sim_100.slcio"])
    
    converter.convert_files(inputfile, output_hdf5_file)
    
    with h5py.File(output_hdf5_file, 'r') as file:
        # Check there is at least one event group
        assert "evt_0" in file, "Expected 'evt_0' group missing"
        
        # For that event, check it has the three main subdetector collections
        for col in ["ECalColls", "HCalColls", "TrkHitColls"]:
            assert "evt_0/"+col in file, "Expected subdetector collection 'evt_0/'"+col+"' missing"
        
        # check for all expected collections in the ECalCols and hit level datasets
        hit_vars = ["hit_times", "hit_energy", "hit_cell_id", "hit_x_pos", "hit_y_pos", "hit_z_pos"]
        ec_colls = ["ECalBarrelCollection", "ECalEndcapCollection"]
        for col in ec_colls:
            assert "evt_0/ECalColls/"+col in file, "Expected ECal collection 'evt_0/ECalColls/"+col+"' missing"
            # check for all the hit-level datasets
            for var in hit_vars:
                assert "evt_0/ECalColls/"+col+"/"+var in file, "Expected hit-level dataset 'evt_0/ECalColls/"+col+"/"+var+"' missing"



