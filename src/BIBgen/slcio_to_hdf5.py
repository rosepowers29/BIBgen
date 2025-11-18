from pyLCIO import IOIMPL, EVENT, UTIL
import h5py
from argparse import ArgumentParser
import numpy as np

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

def make_event_group(hdf5_file, evt_num):
    """
    Create a group one level below file to represent a single event.

    Arguments:
        hdf5_file (h5py file object): the hdf5 file we open before the event loop, where we will write all collections
        evt_num (int): the event currently being iterated over

    Returns:
        event_group (h5py group object): the "group" or sub-order of data representing the event, which will contain hit collections.
    """
    event_group_name = "evt_"+str(evt_num)
    event_group = hdf5_file.create_group(event_group_name)
    return event_group

def make_collection_groups(hdf5_evt):
    """
    Create two levels of hdf5 subgroups for each event "group", creating this structure:
                    [Evt]

        ____________/ | \____________
       /              |              \ 
      /               |               \ 
     /                |                \ 
    [ECalCols]    [HCalCols]      [TrkHitCols]
    |      \          /\              /    |
    |       \        /  \            /     |

    col_1...col_n col_1...col_n   col_1...col_n
    

    Arguments:
        hdf5_evt (h5py group object): an event-level (first subgroup below file) group. For instance,
                                  if we want to look at the first event in file "f", hdf5_evt could be f['evt_0'].

    Returns:
        ec_dict (dictionary): a dict of h5py group objects two levels below file, each value corresponding to each ECal hit collection (see diagram above), each key is the name of the collection
        
        hc_dict (dictionary): a dict of h5py group objects two levels below file, each value corresponding to each HCal hit collection (see diagram above), each key is the name of the collection
        
        th_dict (dictionary): a dict of h5py group objects two levels below file, each value corresponding to each tracker hit collection (see diagram above), each key is the name of the collection
    """
    # first make new subgroups for ecal, hcal, and tracker hits
    ec_group = hdf5_evt.create_group("ECalColls")
    hc_group = hdf5_evt.create_group("HCalColls")
    trkhit_group = hdf5_evt.create_group("TrkHitColls")
    # these are the names of the SLCIO collections
    ec_colls = ["ECalBarrelCollection", "ECalEndcapCollection"]
    hc_colls = ["HCalBarrelCollection", "HCalEndcapCollection"]
    trk_hit_colls = ["InnerTrackerBarrelCollection", "InnerTrackerEndcapCollection", "OuterTrackerBarrelCollection", 
             "OuterTrackerEndcapCollection", "VertexBarrelCollection", "VertexEndcapCollection"]
    ec_dict = {}
    hc_dict = {}
    th_dict = {}
    for coll in ec_colls:
        # make new sub-subgroups for each collection, store them in a dict
        ec_dict[coll] = ec_group.create_group(coll)
    for coll in hc_colls:
        hc_dict[coll] = hc_group.create_group(coll)
    for coll in trk_hit_colls:
        th_dict[coll] = trkhit_group.create_group(coll)
    return ec_dict, hc_dict, th_dict

def fill_calohit_datasets(slcio_coll, hdf5_coll):
    """
    Create an array-like "dataset" at the lowest data level (below subdetector collections) for each calo hit variable of interest.
    
    Arguments:
        slcio_coll (LCCollection object): the calohit collection in SLCIO format we want to translate.
        hdf5_coll (h5py group object): the calohit collection "group" in the h5py file, created with make_collection_groups(), 
        where we want to write the datasets.

    Returns: 
        hit_sets (dictionary): a dict of h5py "dataset" objects for each variable of interest

    Usage note: slcio_coll and hdf5_coll should correspond to the same collection. For example, for the ecal barrel hits,
    one might call fill_calohit_datasets(event.getCollection("ECalBarrelCollection"), file['evt_0/EcalColls/ECalBarrelCollection'])
    to write the first event's ECalBarrelCollection to the hdf5 file.
    
    """
    hit_times = []
    hit_energy = []
    hit_cell_id = []
    hit_x_pos = []
    hit_y_pos = []
    hit_z_pos = []

    for hit in slcio_coll:
        try:
            hit_times.append(hit.getTime())
        except Exception:
            hit_times.append(-1.)
        hit_energy.append(hit.getEnergy())
        hit_cell_id.append(hit.getCellID0())
        hit_x_pos.append(hit.getPosition()[0])
        hit_y_pos.append(hit.getPosition()[1])
        hit_z_pos.append(hit.getPosition()[2])

    hit_sets = {"hit_times": np.array(hit_times), "hit_energy": np.array(hit_energy), "hit_cell_id": np.array(hit_cell_id), 
                "hit_x_pos": np.array(hit_x_pos), "hit_y_pos": np.array(hit_y_pos), "hit_z_pos": np.array(hit_z_pos)}

    for key in hit_sets.keys():
        hit_sets[key] = hdf5_coll.create_dataset(key, data = hit_sets[key]) # replace arrays with hdf5 datasets
    
    return hit_sets

def fill_trackerhit_datasets(slcio_coll, hdf5_coll):
    """
    Create an array-like "dataset" at the lowest data level (below subdetector collections) for each tracker hit variable of interest.
    
    Arguments:
        slcio_coll (LCCollection object): the calohit collection in SLCIO format we want to translate.
        hdf5_coll (h5py group object): the calohit collection "group" in the h5py file, created with make_collection_groups(), 
        where we want to write the datasets.

    Returns: 
        hit_sets (dictionary): a dict of h5py "dataset" objects for each variable of interest

    Note: The only difference between fill_trackerhit_datasets and fill_calohit_datasets is the energy variable,
    which is hit_e_dep for tracker hits and hit_energy for calohits. If a calohitcollection is passed to fill_trackerhit_datasets
    or vice versa, an AttributeError will result.
    """
    hit_times = []
    hit_e_dep = []
    hit_cell_id = []
    hit_x_pos = []
    hit_y_pos = []
    hit_z_pos = []

    for hit in slcio_coll:
        try:
            hit_times.append(hit.getTime())
        except Exception:
            hit_times.append(-1.)
        hit_e_dep.append(hit.getEDep())
        hit_cell_id.append(hit.getCellID0())
        hit_x_pos.append(hit.getPosition()[0])
        hit_y_pos.append(hit.getPosition()[1])
        hit_z_pos.append(hit.getPosition()[2])

    hit_sets = {"hit_times": np.array(hit_times), "hit_e_dep": np.array(hit_e_dep), "hit_cell_id": np.array(hit_cell_id), 
                "hit_x_pos": np.array(hit_x_pos), "hit_y_pos": np.array(hit_y_pos), "hit_z_pos": np.array(hit_z_pos)}

    for key in hit_sets.keys():
        hit_sets[key] = hdf5_coll.create_dataset(key, data = hit_sets[key]) # replace arrays with hdf5 datasets
    return hit_sets




def main():
    """
    The main routine. Called when "python slcio_to_hdf5.py -i [input_files] is entered at command line.
    Grabs slcio files from the commandline, opens them, gets desired collections, and writes to hdf5 file.

    Arguments: None
    
    Returns: None

    """
    files_to_read, file_to_write = parse_input()
    
    # create hdf5 file before event loop
    with h5py.File(file_to_write, 'w') as hdf_file:

        # initialize event iter; we will consider all events in all files
        evt_iter = 0
        file_iter = 0
        for file in files_to_read:
            # read in the LCIO file
            reader = IOIMPL.LCFactory.getInstance().createLCReader()
            try:
                reader.open(file)
            except Exception:
                # let it skip any "bad" (corrupted) files without breaking
                print(f"skipping file {file_iter}")
                file_iter += 1
                continue
            # start event loop
            for event in reader:
                # make the hdf5 event
                hdf_evt = make_event_group(hdf_file, evt_iter)
                
                # loop over collections and make the collection groups
                ec_dict, hc_dict, th_dict = make_collection_groups(hdf_evt)
                
                # see which collections are available in the event
                collections_avail = event.getCollectionNames()

                # fill the datasets for calorimeter hits and tracker hits for this event
                for ecal_coll in ec_dict.keys():
                    if ecal_coll not in collections_avail:
                        continue
                    fill_calohit_datasets(event.getCollection(ecal_coll), ec_dict[ecal_coll])
                for hcal_coll in hc_dict.keys():
                    if hcal_coll not in collections_avail:
                        continue
                    fill_calohit_datasets(event.getCollection(hcal_coll), hc_dict[hcal_coll])
                for th_coll in th_dict.keys():
                    if th_coll not in collections_avail:
                        continue
                    fill_trackerhit_datasets(event.getCollection(th_coll), th_dict[th_coll])

                evt_iter += 1
                # end of event loop
            file_iter += 1
            reader.close()
            # end file loop



if __name__ == "__main__":
    main()
