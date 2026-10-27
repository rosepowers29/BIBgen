#!/bin/bash
# Slurm (Perlmutter) counterpart of submit_train.sub (OSG) / train.slurm (Della): trains one model.
#
# Normally submitted by submit_train_perlmutter.sh, which reads rows of an experiments file and
# passes each row, plus REPO/DATA_DIR/ACCOUNT from the file's settings, in through the
# environment. To submit a single run by hand, from $REPO/examples/training/:
#
#   mkdir -p logs
#   REPO=$PSCRATCH/BIBgen DATA=diffused_cyl_phipi4_large_logE.hdf5 \
#   CONFIG=equivariant_denoiser.json TAG=v12 SCHEDULE=noise_schedule.csv \
#   sbatch --account=mxyz --job-name=train_v12 --mail-user=me@princeton.edu --mail-type=END,FAIL \
#   train_perlmutter.slurm
#
# Command-line sbatch options override the #SBATCH defaults below, e.g. --time=24:00:00.
#
# Defaults to the `shared` QOS, which bills per-GPU rather than per-node (each Perlmutter GPU
# node has 4 A100s) -- the right choice for a single-GPU job like this one. Cores scale with the
# GPU fraction requested: 1 GPU gets 1/4 of a node's 128 physical cores, so --gpus-per-task=1
# pairs with -c 32 below. To run across multiple full nodes instead (e.g. a distributed/
# data-parallel job), switch to
#   --qos=regular --nodes=N --ntasks=<N*4> --gpus-per-task=1 -c 32
# and change the srun line below to request <N*4> tasks and GPUs. Always keep -C gpu (Perlmutter
# has no separate GPU partition or --constraint=gpu40 like Della -- that is a Della-only name).

#SBATCH --job-name=train
#SBATCH --output=logs/%x_%j.out
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00

set -euo pipefail

# Inside a batch job the script runs from a spool copy, so the repo cannot be found
# relative to $0. It comes from REPO= in the experiments file, via the submit wrapper.
: "${REPO:?REPO (path to the BIBgen checkout to run) is not set}"
DATA_DIR=${DATA_DIR:-$REPO/data}
OUT_DIR=${OUT_DIR:-$REPO/examples/training}
EPOCHS=${EPOCHS:-151}
BATCH_SIZE=${BATCH_SIZE:-}
TRAIN_ARGS=${TRAIN_ARGS:-}

# train.py requires exactly one of -b/--batch-size or --hit-budget. Omit -b entirely when
# BATCH_SIZE isn't set, rather than defaulting it, so --hit-budget can be passed via TRAIN_ARGS
# instead without colliding with it.
batch_opts=()
[[ -n $BATCH_SIZE ]] && batch_opts=(-b "$BATCH_SIZE")

: "${DATA:?DATA (diffused hdf5, name in DATA_DIR or a path) is not set}"
: "${CONFIG:?CONFIG (model json in config/models/) is not set}"
: "${TAG:?TAG is not set}"
: "${SCHEDULE:?SCHEDULE (noise schedule csv in config/noise_schedules/) is not set}"

# A bare filename is looked up in DATA_DIR; anything with a slash is used as given.
[[ $DATA == */* ]] && data_path=$DATA || data_path=$DATA_DIR/$DATA

if [[ ! -x $REPO/.venv/bin/python ]]; then
    echo "No environment at $REPO/.venv -- run 'uv sync' on a login node first." >&2
    exit 1
fi
source "$REPO/.venv/bin/activate"
export PYTHONUNBUFFERED=1

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "=== tag=$TAG data=$data_path config=$CONFIG schedule=$SCHEDULE epochs=$EPOCHS batch=${BATCH_SIZE:-<hit-budget>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$OUT_DIR"
# shellcheck disable=SC2086  # TRAIN_ARGS is deliberately word-split
srun -n 1 -c 32 --cpu-bind=cores -G 1 --gpu-bind=none \
    python "$REPO/examples/training/train.py" "$data_path" \
    "$REPO/config/noise_schedules/$SCHEDULE" "$REPO/config/models/$CONFIG" \
    -e "$EPOCHS" "${batch_opts[@]}" -t "$TAG" $TRAIN_ARGS

echo "=== finished at $(date)"