import argparse
import warnings

import h5py
import numpy as np

from BIBgen.preprocessing import Sphering

def build_unsphered(event_id, gather, use_cylindrical, min_phi, max_phi, log_energy):
    e_raw = gather(event_id, "hit_energy")
    x_raw = gather(event_id, "hit_x_pos")
    y_raw = gather(event_id, "hit_y_pos")
    z_raw = gather(event_id, "hit_z_pos")

    if log_energy:
        e_raw = np.log(e_raw)

    if use_cylindrical:
        phi_raw = np.arctan2(y_raw, x_raw)
        s_raw = np.sqrt(x_raw**2 + y_raw**2)
        return np.stack([e_raw, phi_raw, s_raw, z_raw], axis=-1)[(phi_raw <= max_phi) & (phi_raw >= min_phi)]
    return np.stack([e_raw, x_raw, y_raw, z_raw], axis=-1)

def main(args):
    mm_path = args.mm_path
    mp_path = args.mp_path
    outpath = args.out
    data_split = args.split.split(",")
    use_cylindrical = args.cylindrical
    max_phi = args.phi_window
    min_phi = -args.phi_window
    log_energy = not args.raw_energy   # default: ln(E); pass --raw-energy for raw E
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
        mpfile["{}/ECalColls/ECalBarrelCollection/{}".format(ev, key)],
    ])

    nevents = len(mmfile.keys())
    ntrain = int(data_split[0])
    nval = int(data_split[1])
    ntest = int(data_split[2])

    train_unsphered = []
    for ievent in range(ntrain):
        event_id = "evt_{}".format(ievent)
        train_unsphered.append(build_unsphered(event_id, gather, use_cylindrical, min_phi, max_phi, log_energy))
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
    sphere_group.attrs["log_energy"] = log_energy

    del train
    del train_unsphered

    val_group = outfile.create_group("validation")
    for ievent in range(ntrain, ntrain+nval):
        event_id = "evt_{}".format(ievent)
        unsphered = build_unsphered(event_id, gather, use_cylindrical, min_phi, max_phi, log_energy)
        sphered = sphering.transform(unsphered)

        event_group = val_group.create_group(event_id)
        event_group.create_dataset("tau0", data=sphered)
        print("Processed {} for validation".format(event_id))

    test_group = outfile.create_group("test")
    for ievent in range(ntrain+nval, ntrain+nval+ntest):
        event_id = "evt_{}".format(ievent)
        unsphered = build_unsphered(event_id, gather, use_cylindrical, min_phi, max_phi, log_energy)

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
    parser.add_argument("-c", "--cylindrical", action="store_true", help="Whether the training data should be in cylindrical coordinates")
    parser.add_argument("-p", "--phi-window", default=np.pi, type=float, help="Phi to slice the data.")
    parser.add_argument("--raw-energy", action="store_true", help="Store raw E instead of the default ln(E)")
    print("\nFinished with exit code:", main(parser.parse_args()))
