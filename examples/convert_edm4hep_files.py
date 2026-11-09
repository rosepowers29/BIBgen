from argparse import ArgumentParser
from podio import root_io
import h5py
import numpy as np
import BIBgen.edm4hep_to_hdf5 as converter

def parse_input():
    """
    Unpack the commandline input and return a list of files for reading.

    Notes: Supports multiple file inputs

    Arguments:
        None (to be called first in function __main__)

    Returns:
        edm4hep_files (list): A list of edm4hep.root files
        hdf5_file (string): The name of the output file
        calo_time_mode (string): how to derive each calo hit's time from its CaloHitContributions
        correct_for_propagation (bool): whether to apply a time-of-flight correction to each calo hit contribution
    """
    edm4hep_files = []
    parser = ArgumentParser()
    parser.add_argument('-i', '--inFiles', dest = 'inFiles', help= 'edm4hep.root files to translate', nargs='+')
    parser.add_argument('-o', '--outFile', dest = 'outFile', help = 'output hdf file, with .hdf5 extension')
    parser.add_argument('--caloTimeMode', dest = 'caloTimeMode', default = 'earliest',
                         choices = converter.CALO_TIME_MODES,
                         help = "how to reduce each SimCalorimeterHit's CaloHitContributions to a single hit_time "
                                "(default: earliest)")
    parser.add_argument('--correctForPropagation', dest = 'correctForPropagation', action = 'store_true',
                         help = "Subtract the time of flight from the origin to each calo hit's position "
                                "(|r| / c_light) from every CaloHitContribution's time before reducing "
                                "(default: off, i.e. raw contribution times)")
    args = parser.parse_args()
    for file in args.inFiles:
        edm4hep_files.append(file)
    return edm4hep_files, args.outFile, args.caloTimeMode, args.correctForPropagation

if __name__ == "__main__":
    files_to_read, file_to_write, calo_time_mode, correct_for_propagation = parse_input()
    converter.convert_files(files_to_read, file_to_write, calo_time_mode = calo_time_mode,
                             correct_for_propagation = correct_for_propagation)
