import h5py
import numpy as np

# Hits outside one bunch crossing are out-of-time background.
BUNCH_CROSSING_MIN_TIME = 0.0  # ns
BUNCH_CROSSING_MAX_TIME = 3.3e4  # ns

def filter_file(input_path, output_path, min_time = BUNCH_CROSSING_MIN_TIME, max_time = BUNCH_CROSSING_MAX_TIME):
    """
    Post-translation time filter: copy a converted hdf5 file (edm4hep_to_hdf5/slcio_to_hdf5 output)
    to a new file, dropping hits whose hit_time falls outside [min_time, max_time] from every hit
    collection. All datasets in a collection are masked together, so they stay aligned hit-for-hit.

    Arguments:
        input_path (str): path to an existing converted hdf5 file.
        output_path (str): path to write the filtered hdf5 file. Must differ from input_path,
            since datasets are read from the input file while writing the output file.
        min_time (float): minimum hit_time (inclusive) to keep, in ns.
        max_time (float): maximum hit_time (inclusive) to keep, in ns.

    Returns: None
    """
    with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
        for evt_name, evt_group in fin.items():
            out_evt = fout.create_group(evt_name)
            for subdet_name, subdet_group in evt_group.items():
                out_subdet = out_evt.create_group(subdet_name)
                for coll_name, coll_group in subdet_group.items():
                    out_coll = out_subdet.create_group(coll_name)
                    hit_times = coll_group["hit_times"][:]
                    mask = (hit_times >= min_time) & (hit_times <= max_time)
                    for var_name, dataset in coll_group.items():
                        out_coll.create_dataset(var_name, data = dataset[:][mask])
