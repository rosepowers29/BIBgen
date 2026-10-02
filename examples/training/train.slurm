#!/bin/bash
# Slurm (Della) counterpart of submit_train.sub + run_train.sh: trains one model.
#
# Normally submitted by submit_train.sh, which reads rows of an experiments file and passes
# each row, plus REPO/DATA_DIR from the file's settings, in through the environment. To submit
# a single run by hand, from $REPO/examples/training/:
#
#   mkdir -p logs
#   REPO=/scratch/gpfs/GROUP/me/BIBgen DATA=diffused_cyl_phipi4_large_logE.hdf5 \
#   CONFIG=equivariant_denoiser.json TAG=v12 SCHEDULE=noise_schedule.csv \
#   sbatch --job-name=train_v12 --mail-user=me@princeton.edu --mail-type=END,FAIL train.slurm
#
# Command-line sbatch options override the #SBATCH defaults below, e.g. --time=48:00:00.
#
# The defaults are sized from the first full run (della_v1, job 14781631: 151 epochs of
# equivariant_denoiser.json on diffused_cyl_phipi4_large_logE.hdf5, batch 5):
#   time  8:57 on a 40 GB MIG A100 (~3.5 min/epoch) -> 12 h leaves ~35% headroom
#   CPU   1.27 cores busy on average -> 2 cores. The Python loop is single-threaded, but
#         PyTorch/CUDA and h5py run helper threads beside it, so 1 core would contend.
#   mem   1.6 GB peak -> 4 GB
#   GPU   6.3 GB of GPU memory. That would fit a 10 GB MIG, but with ~1/3 of this slice's
#         compute and only 1 core, so the run would likely exceed 24 h there; gpu40 stays.
# Rescale --time if you change epochs, batch size, model size or dataset.
# To run on a MIG GPU (1/7 of an A100, 10 GB GPU memory; usually no queue) instead, add
#   --partition=mig --constraint= --cpus-per-task=1
# The empty --constraint= clears gpu40 below, which MIG nodes do not satisfy, and MIG
# allows only 1 core. Never pass
# --partition=gpu: Della rejects it.

#SBATCH --job-name=train
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu40
#SBATCH --time=12:00:00

set -euo pipefail

# Inside a batch job the script runs from a spool copy, so the repo cannot be found
# relative to $0. It comes from REPO= in the experiments file, via the submit wrapper.
: "${REPO:?REPO (path to the BIBgen checkout to run) is not set}"
DATA_DIR=${DATA_DIR:-$REPO/data}
OUT_DIR=${OUT_DIR:-$REPO/examples/training}
EPOCHS=${EPOCHS:-151}
BATCH_SIZE=${BATCH_SIZE:-5}
TRAIN_ARGS=${TRAIN_ARGS:-}

: "${DATA:?DATA (diffused hdf5, name in DATA_DIR or a path) is not set}"
: "${CONFIG:?CONFIG (model json in config/) is not set}"
: "${TAG:?TAG is not set}"
: "${SCHEDULE:?SCHEDULE (noise schedule csv in config/) is not set}"

# A bare filename is looked up in DATA_DIR; anything with a slash is used as given.
[[ $DATA == */* ]] && data_path=$DATA || data_path=$DATA_DIR/$DATA

if [[ ! -x $REPO/.venv/bin/python ]]; then
    echo "No environment at $REPO/.venv -- run 'uv sync' on a login node first." >&2
    exit 1
fi
source "$REPO/.venv/bin/activate"
export PYTHONUNBUFFERED=1

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "=== tag=$TAG data=$data_path config=$CONFIG schedule=$SCHEDULE epochs=$EPOCHS batch=$BATCH_SIZE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$OUT_DIR"
# shellcheck disable=SC2086  # TRAIN_ARGS is deliberately word-split
python "$REPO/examples/training/train.py" "$data_path" \
    "$REPO/config/$SCHEDULE" "$REPO/config/$CONFIG" \
    -e "$EPOCHS" -b "$BATCH_SIZE" -t "$TAG" $TRAIN_ARGS

echo "=== finished at $(date)"
