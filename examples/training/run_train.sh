#!/bin/bash
# Condor counterpart of train.slurm (Della) / train_perlmutter.slurm: trains one model.
# Positional args (set by submit_train.sub's `arguments` line, from a row of the experiments
# file): data config tag schedule batch_size. EPOCHS/TRAIN_ARGS come through as environment
# variables instead (via submit_train.sub's `environment` line), since Condor's queue-from-file
# substitution doesn't word-split a column's value the way a positional argument would.

export PYTHONPATH=$PYTHONPATH:/opt/conda/lib/python3.10/site-packages
pip install h5py

export PYTHONPATH=$PWD/src:$PYTHONPATH

data=$1
config=$2
tag=$3
schedule=$4
batch_size=$5
EPOCHS=${EPOCHS:-151}
TRAIN_ARGS=${TRAIN_ARGS:-}

# train.py requires exactly one of -b/--batch-size or --hit-budget. Omit -b entirely when
# batch_size isn't set, rather than defaulting it, so --hit-budget can be passed via TRAIN_ARGS
# instead without colliding with it.
batch_opts=()
[[ -n $batch_size ]] && batch_opts=(-b "$batch_size")

# shellcheck disable=SC2086  # TRAIN_ARGS is deliberately word-split
python train.py "$data" "config/noise_schedules/$schedule" "config/models/$config" \
    -e "$EPOCHS" "${batch_opts[@]}" -t "$tag" $TRAIN_ARGS
