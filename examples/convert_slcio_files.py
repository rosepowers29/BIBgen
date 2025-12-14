from argparse import ArgumentParser
from pyLCIO import IMPL, EVENT, UTIL
import h5py
import numpy as np
import BIBgen.slcio_to_hdf5 as converter

def parse_input():
    """
    Unpack the commandline input and return a list of files for reading.

    Notes: Supports multiple file inputs

    Arguments:
        None (to be called first in function __main__)

    Returns: 
        slcio_files (list): A list of slcio files
        hdf5_file (string): The name of the output file
    """
    slcio_files = []
    parser = ArgumentParser()
    parser.add_argument('-i', '--inFiles', dest = 'inFiles', help= 'slcio files to translate', nargs='+')
    parser.add_argument('-o', '--outFile', dest = 'outFile', help = 'output hdf file, with .hdf5 extension')
    args = parser.parse_args()
    for file in args.inFiles:
        slcio_files.append(file)
    return slcio_files, args.outFile

if __name__ == "__main__":
    files_to_read, file_to_write = parse_input()
    converter.convert_files(files_to_read, file_to_write)
