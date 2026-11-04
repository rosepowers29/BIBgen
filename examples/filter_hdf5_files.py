from argparse import ArgumentParser
import os
import BIBgen.filter_hdf5 as filterer

def parse_input():
    """
    Unpack the commandline input.

    Notes: Supports multiple file inputs. Each input file is filtered independently into its
    own output file (no merging).

    Arguments:
        None (to be called first in function __main__)

    Returns:
        in_files (list): converted hdf5 files to filter
        out_files (list): where to write each filtered file, same length/order as in_files
        min_time (float): minimum hit_time (ns) to keep
        max_time (float): maximum hit_time (ns) to keep
    """
    parser = ArgumentParser()
    parser.add_argument('-i', '--inFiles', dest = 'inFiles', help = 'converted hdf5 files to filter', nargs = '+', required = True)
    parser.add_argument('-o', '--outFiles', dest = 'outFiles', nargs = '+',
                         help = 'output filtered hdf5 files, one per input file, in the same order. '
                                'If omitted, each is written alongside its input as "<name>_filtered.hdf5", '
                                'or into --outDir if given.')
    parser.add_argument('--outDir', dest = 'outDir', default = None,
                         help = 'directory to write "<name>_filtered.hdf5" outputs into, when --outFiles is not given')
    parser.add_argument('--minTime', dest = 'minTime', type = float, default = filterer.BUNCH_CROSSING_MIN_TIME,
                         help = f'minimum hit_time in ns to keep (default: {filterer.BUNCH_CROSSING_MIN_TIME})')
    parser.add_argument('--maxTime', dest = 'maxTime', type = float, default = filterer.BUNCH_CROSSING_MAX_TIME,
                         help = f'maximum hit_time in ns to keep (default: {filterer.BUNCH_CROSSING_MAX_TIME})')
    args = parser.parse_args()

    if args.outFiles:
        if len(args.outFiles) != len(args.inFiles):
            parser.error('--outFiles must list exactly one output path per input file')
        out_files = args.outFiles
    else:
        out_dir = args.outDir
        out_files = []
        for in_file in args.inFiles:
            stem, ext = os.path.splitext(os.path.basename(in_file))
            out_name = f"{stem}_filtered{ext}"
            out_files.append(os.path.join(out_dir, out_name) if out_dir else os.path.join(os.path.dirname(in_file), out_name))

    return args.inFiles, out_files, args.minTime, args.maxTime

if __name__ == "__main__":
    in_files, out_files, min_time, max_time = parse_input()
    for in_file, out_file in zip(in_files, out_files):
        print(f"filtering {in_file} -> {out_file}")
        filterer.filter_file(in_file, out_file, min_time = min_time, max_time = max_time)
