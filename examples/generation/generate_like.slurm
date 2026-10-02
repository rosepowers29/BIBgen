#!/bin/bash
# Slurm (Della) counterpart of submit_generate_like.sub + run_generate_like.sh: generates
# <tag>_like.hdf5 from denoiser_<tag>.pth, reproducing the test-event sizes in SIZES.
#
# Normally submitted by submit_generate_like.sh, which passes REPO from the experiments file.
# To submit a single run by hand, from $REPO/examples/generation/:
#
#   mkdir -p logs
#   REPO=/scratch/gpfs/GROUP/me/BIBgen CONFIG=equivariant_denoiser.json TAG=v12 \
#   SCHEDULE=noise_schedule.csv \
#   sbatch --job-name=generate_v12 --mail-user=me@princeton.edu --mail-type=END,FAIL generate_like.slurm
#
# Defaults to a MIG GPU (1/7 of an A100, 10 GB GPU memory, 1 core, fixed 32 GB CPU memory),
# which Della asks for whenever a job fits. To use a full 40 GB A100 instead, add
#   --partition= --constraint=gpu40 --mem=16G
# to the sbatch command line. The empty --partition= is deliberate: it clears `mig` below, and
# Della rejects an explicit --partition=gpu (GPU jobs are routed there automatically).

#SBATCH --job-name=generate
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=mig
#SBATCH --time=06:00:00

set -euo pipefail

# Inside a batch job the script runs from a spool copy, so the repo cannot be found
# relative to $0. It comes from REPO= in the experiments file, via the submit wrapper.
: "${REPO:?REPO (path to the BIBgen checkout to run) is not set}"
MODEL_DIR=${MODEL_DIR:-$REPO/examples/training}
OUT_DIR=${OUT_DIR:-$REPO/examples/generation}
SIZES=${SIZES:-$REPO/examples/generation/test_sizes_large.csv}
GENERATE_ARGS=${GENERATE_ARGS:-}

: "${CONFIG:?CONFIG (model json in config/) is not set}"
: "${TAG:?TAG is not set}"
: "${SCHEDULE:?SCHEDULE (noise schedule csv in config/, same as training) is not set}"

model_path=$MODEL_DIR/denoiser_$TAG.pth
for f in "$model_path" "$SIZES"; do
    [[ -f $f ]] || { echo "Missing input: $f" >&2; exit 1; }
done

if [[ ! -x $REPO/.venv/bin/python ]]; then
    echo "No environment at $REPO/.venv -- run 'uv sync' on a login node first." >&2
    exit 1
fi
source "$REPO/.venv/bin/activate"
export PYTHONUNBUFFERED=1

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "=== tag=$TAG model=$model_path config=$CONFIG schedule=$SCHEDULE sizes=$SIZES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$OUT_DIR"
# shellcheck disable=SC2086  # GENERATE_ARGS is deliberately word-split
python "$REPO/examples/generation/generate_like.py" "$model_path" \
    "$REPO/config/$CONFIG" "$REPO/config/$SCHEDULE" "$SIZES" -t "$TAG" $GENERATE_ARGS

echo "=== finished at $(date)"
