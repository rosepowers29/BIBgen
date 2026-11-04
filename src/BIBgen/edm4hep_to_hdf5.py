from podio import root_io
import h5py
import numpy as np

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
    # these are the names of the edm4hep collections
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

CALO_TIME_MODES = ("earliest", "energy_weighted", "highest_energy")

# Hits outside one bunch crossing are out-of-time background and are dropped at conversion.
BUNCH_CROSSING_MIN_TIME = 0.0  # ns
BUNCH_CROSSING_MAX_TIME = 3.3e4  # ns

def filter_hits_to_bunch_crossing(hit_sets):
    """
    Drop hits whose hit_time falls outside [BUNCH_CROSSING_MIN_TIME, BUNCH_CROSSING_MAX_TIME],
    applying the same mask to every array in hit_sets so they stay aligned hit-for-hit.

    Arguments:
        hit_sets (dict[str, np.ndarray]): per-hit arrays (hit_times plus the other hit variables),
            all the same length, as built by fill_calohit_datasets/fill_trackerhit_datasets.

    Returns:
        hit_sets (dict[str, np.ndarray]): same keys, each array masked to the bunch crossing window.
    """
    mask = (hit_sets["hit_times"] >= BUNCH_CROSSING_MIN_TIME) & (hit_sets["hit_times"] <= BUNCH_CROSSING_MAX_TIME)
    return {key: val[mask] for key, val in hit_sets.items()}

def get_calo_hit_time(hit, time_mode = "earliest"):
    """
    Derive a single hit time for a SimCalorimeterHit from its CaloHitContributions.

    Unlike LCIO's SimCalorimeterHit, edm4hep's SimCalorimeterHit has no getTime() of its own;
    its contributions (edm4hep::CaloHitContribution) each carry a time and an energy, and we
    have to reduce those down to one value per hit.

    Arguments:
        hit (edm4hep::SimCalorimeterHit): the calo hit whose contributions we want to reduce.
        time_mode (str): one of CALO_TIME_MODES:
            "earliest": the time of the earliest (minimum-time) contribution
            "energy_weighted": the energy-weighted average time over all contributions
            "highest_energy": the time of the contribution that deposited the most energy

    Returns:
        time (float): the derived hit time, or -1. if the hit has no contributions.
    """
    if time_mode not in CALO_TIME_MODES:
        raise ValueError(f"Unknown calo hit time_mode '{time_mode}', expected one of {CALO_TIME_MODES}")

    times = []
    energies = []
    for contribution in hit.getContributions():
        times.append(contribution.getTime())
        energies.append(contribution.getEnergy())

    if not times:
        return -1.

    if time_mode == "earliest":
        return min(times)
    elif time_mode == "energy_weighted":
        total_energy = sum(energies)
        if total_energy == 0:
            return min(times)
        return sum(t * e for t, e in zip(times, energies)) / total_energy
    else: # "highest_energy"
        return times[energies.index(max(energies))]

def fill_calohit_datasets(edm4hep_coll, hdf5_coll, time_mode = "earliest"):
    """
    Create an array-like "dataset" at the lowest data level (below subdetector collections) for each calo hit variable of interest.

    Arguments:
        edm4hep_coll (edm4hep::SimCalorimeterHitCollection): the calohit collection in edm4hep format we want to translate.
        hdf5_coll (h5py group object): the calohit collection "group" in the h5py file, created with make_collection_groups(),
        where we want to write the datasets.
        time_mode (str): how to reduce each hit's CaloHitContributions down to a single hit_time. See get_calo_hit_time().

    Returns:
        hit_sets (dictionary): a dict of h5py "dataset" objects for each variable of interest

    Usage note: edm4hep_coll and hdf5_coll should correspond to the same collection. For example, for the ecal barrel hits,
    one might call fill_calohit_datasets(frame.get("ECalBarrelCollection"), file['evt_0/EcalColls/ECalBarrelCollection'])
    to write the first event's ECalBarrelCollection to the hdf5 file.

    """
    hit_times = []
    hit_energy = []
    hit_cell_id = []
    hit_x_pos = []
    hit_y_pos = []
    hit_z_pos = []

    for hit in edm4hep_coll:
        hit_times.append(get_calo_hit_time(hit, time_mode = time_mode))
        hit_energy.append(hit.getEnergy())
        hit_cell_id.append(hit.getCellID())
        pos = hit.getPosition()
        hit_x_pos.append(pos.x)
        hit_y_pos.append(pos.y)
        hit_z_pos.append(pos.z)

    hit_sets = {"hit_times": np.array(hit_times), "hit_energy": np.array(hit_energy), "hit_cell_id": np.array(hit_cell_id),
                "hit_x_pos": np.array(hit_x_pos), "hit_y_pos": np.array(hit_y_pos), "hit_z_pos": np.array(hit_z_pos)}
    hit_sets = filter_hits_to_bunch_crossing(hit_sets)

    for key in hit_sets.keys():
        hit_sets[key] = hdf5_coll.create_dataset(key, data = hit_sets[key]) # replace arrays with hdf5 datasets

    return hit_sets

def fill_trackerhit_datasets(edm4hep_coll, hdf5_coll):
    """
    Create an array-like "dataset" at the lowest data level (below subdetector collections) for each tracker hit variable of interest.

    Arguments:
        edm4hep_coll (edm4hep::SimTrackerHitCollection): the calohit collection in edm4hep format we want to translate.
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

    for hit in edm4hep_coll:
        try:
            hit_times.append(hit.getTime())
        except Exception:
            hit_times.append(-1.)
        hit_e_dep.append(hit.getEDep())
        hit_cell_id.append(hit.getCellID())
        pos = hit.getPosition()
        hit_x_pos.append(pos.x)
        hit_y_pos.append(pos.y)
        hit_z_pos.append(pos.z)

    hit_sets = {"hit_times": np.array(hit_times), "hit_e_dep": np.array(hit_e_dep), "hit_cell_id": np.array(hit_cell_id),
                "hit_x_pos": np.array(hit_x_pos), "hit_y_pos": np.array(hit_y_pos), "hit_z_pos": np.array(hit_z_pos)}
    hit_sets = filter_hits_to_bunch_crossing(hit_sets)

    for key in hit_sets.keys():
        hit_sets[key] = hdf5_coll.create_dataset(key, data = hit_sets[key]) # replace arrays with hdf5 datasets
    return hit_sets




def convert_files(files_to_read, file_to_write, calo_time_mode = "earliest"):
    """
    The main routine. In the executable script, get the arguments from the command line and call this function.
    Grabs edm4hep.root files, opens them, gets desired collections, and writes to hdf5 file.

    Arguments:
        files_to_read (list[str]): list of edm4hep.root filepaths
        file_to_write (str): filepath of output file
        calo_time_mode (str): how to derive each SimCalorimeterHit's hit_time from its CaloHitContributions.
            One of CALO_TIME_MODES ("earliest", "energy_weighted", "highest_energy"). See get_calo_hit_time().

    Returns: None

    """

    # create hdf5 file before event loop
    with h5py.File(file_to_write, 'w') as hdf_file:

        # initialize event iter; we will consider all events in all files
        evt_iter = 0
        file_iter = 0
        for file in files_to_read:
            # read in the edm4hep.root file
            try:
                reader = root_io.Reader(file)
                if file_iter%10 == 0:
                    print(f"reading file {file_iter}...")
            except Exception:
                # let it skip any "bad" (corrupted) files without breaking
                print(f"skipping file {file_iter}")
                file_iter += 1
                continue
            # start event loop
            for frame in reader.get("events"):
                # make the hdf5 event
                hdf_evt = make_event_group(hdf_file, evt_iter)

                # loop over collections and make the collection groups
                ec_dict, hc_dict, th_dict = make_collection_groups(hdf_evt)

                # see which collections are available in the event
                collections_avail = frame.getAvailableCollections()

                # fill the datasets for calorimeter hits and tracker hits for this event
                for ecal_coll in ec_dict.keys():
                    if ecal_coll not in collections_avail:
                        continue
                    fill_calohit_datasets(frame.get(ecal_coll), ec_dict[ecal_coll], time_mode = calo_time_mode)
                for hcal_coll in hc_dict.keys():
                    if hcal_coll not in collections_avail:
                        continue
                    fill_calohit_datasets(frame.get(hcal_coll), hc_dict[hcal_coll], time_mode = calo_time_mode)
                for th_coll in th_dict.keys():
                    if th_coll not in collections_avail:
                        continue
                    fill_trackerhit_datasets(frame.get(th_coll), th_dict[th_coll])

                evt_iter += 1
                # end of event loop
            file_iter += 1
            # end file loop
