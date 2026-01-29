import argparse
import warnings

import h5py
import numpy as np

from BIBgen.preprocessing import Sphering

def main(args):
    """
    Example:
    uv run make_training_data.py /scratch/rosep8/BIBgen/src/BIBgen/sim_mm_0_1000.hdf5 /scratch/rosep8/BIBgen/src/BIBgen/sim_mp_0_1000.hdf5 -o raw_cyl_phipi4_medium.hdf5 -s 70,20,10 -c -p 0.785398
    """
    mm_path = args.mm_path
    mp_path = args.mp_path
    outpath = args.out
    data_split = args.split.split(",")
    use_cylindrical = args.cylindrical
    max_phi = args.phi_window
    min_phi = -args.phi_window
    assert mm_path.endswith(".hdf5")
    assert mp_path.endswith(".hdf5")
    assert outpath.endswith(".hdf5")
    assert len(data_split) == 3

    if not use_cylindrical and max_phi != np.pi:
        warnings.warn("Custom phi range not supported for cartesian coordinates at this time.")

    print("Processing {} events".format(sum(int(d) for d in data_split)))

    mmfile = h5py.File(mm_path, "r")
    mpfile = h5py.File(mp_path, "r")
    outfile = h5py.File(outpath, "w")
    gather = lambda ev, key: np.concatenate([
        mmfile["{}/ECalColls/ECalBarrelCollection/{}".format(ev, key)],
        # mmfile["{}/ECalColls/ECalEndcapCollection/{}".format(ev, key)],
        mpfile["{}/ECalColls/ECalBarrelCollection/{}".format(ev, key)],
        # mpfile["{}/ECalColls/ECalEndcapCollection/{}".format(ev, key)],
    ])

    nevents = len(mmfile.keys())
    ntrain = int(data_split[0])
    nval = int(data_split[1])
    ntest = int(data_split[2])

    # Training
    train_unsphered = []
    for ievent in range(ntrain):
        event_id = "evt_{}".format(ievent)

        e_raw = gather(event_id, "hit_energy")
        x_raw = gather(event_id, "hit_x_pos")
        y_raw = gather(event_id, "hit_y_pos")
        z_raw = gather(event_id, "hit_z_pos")

        if use_cylindrical:
            phi_raw = np.arctan2(y_raw, x_raw)
            s_raw = np.sqrt(x_raw**2 + y_raw**2)
            unsphered = np.stack([e_raw, phi_raw, s_raw, z_raw], axis=-1)[(phi_raw <= max_phi) & (phi_raw >= min_phi)]
        else:
            unsphered = np.stack([e_raw, x_raw, y_raw, z_raw], axis=-1)

        train_unsphered.append(unsphered)
        print("Processed {} for training".format(event_id))

    sphering = Sphering.from_spherings([Sphering.from_data(d) for d in train_unsphered])
    train = [sphering.transform(d) for d in train_unsphered]

    train_group = outfile.create_group("training")
    for ievent in range(ntrain):
        event_id = "evt_{}".format(ievent)
        event_group = train_group.create_group(event_id)
        event_group.create_dataset("tau0", data=train[ievent])

    sphere_group = outfile.create_group("transformation")
    sphere_group.create_dataset("mu", data=sphering.mu)
    sphere_group.create_dataset("std", data=sphering.std)

    del train
    del train_unsphered

    # Validation
    val_group = outfile.create_group("validation")
    for ievent in range(ntrain, ntrain+nval):
        event_id = "evt_{}".format(ievent)

        e_raw = gather(event_id, "hit_energy")
        x_raw = gather(event_id, "hit_x_pos")
        y_raw = gather(event_id, "hit_y_pos")
        z_raw = gather(event_id, "hit_z_pos")

        if use_cylindrical:
            phi_raw = np.arctan2(y_raw, x_raw)
            s_raw = np.sqrt(x_raw**2 + y_raw**2)
            unsphered = np.stack([e_raw, phi_raw, s_raw, z_raw], axis=-1)[(phi_raw <= max_phi) & (phi_raw >= min_phi)]
        else:
            unsphered = np.stack([e_raw, x_raw, y_raw, z_raw], axis=-1)

        sphered = sphering.transform(unsphered)

        event_group = val_group.create_group(event_id)
        event_group.create_dataset("tau0", data=sphered)
        print("Processed {} for validation".format(event_id))

    # Test
    test_group = outfile.create_group("test")
    for ievent in range(ntrain+nval, ntrain+nval+ntest):
        event_id = "evt_{}".format(ievent)

        e_raw = gather(event_id, "hit_energy")
        x_raw = gather(event_id, "hit_x_pos")
        y_raw = gather(event_id, "hit_y_pos")
        z_raw = gather(event_id, "hit_z_pos")

        if use_cylindrical:
            phi_raw = np.arctan2(y_raw, x_raw)
            s_raw = np.sqrt(x_raw**2 + y_raw**2)
            unsphered = np.stack([e_raw, phi_raw, s_raw, z_raw], axis=-1)[(phi_raw <= max_phi) & (phi_raw >= min_phi)]
        else:
            unsphered = np.stack([e_raw, x_raw, y_raw, z_raw], axis=-1)

        event_group = test_group.create_group(event_id)
        event_group.create_dataset("tau0", data=unsphered)
        print("Processed {} for test".format(event_id))

    mmfile.close()
    mpfile.close()
    outfile.close()

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run preprocessing on raw data hdf5 files")
    parser.add_argument("mm_path", help="path to mu- data")
    parser.add_argument("mp_path", help="path to mu+ data")
    parser.add_argument("-o", "--out", default="raw_data.hdf5", help="path to output")
    parser.add_argument("-s", "--split", default="700,200,100", help="training,validation,test split")
    parser.add_argument("-c", "--cylindrical", action="store_true")
    parser.add_argument("-p", "--phi-window", default=np.pi, type=float)
    print("\nFinished with exit code:", main(parser.parse_args()))