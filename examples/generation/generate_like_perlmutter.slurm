#!/bin/bash
# Slurm (Perlmutter) counterpart of submit_generate_like.sub (OSG) / generate_like.slurm
# (Della): generates <tag>_like.hdf5 from denoiser_<tag>.pth, reproducing the test-event sizes
# in SIZES.
#
# Normally submitted by submit_generate_like_perlmutter.sh, which passes REPO/ACCOUNT from the
# experiments file. To submit a single run by hand, from $REPO/examples/generation/:
#
#   mkdir -p logs
#   REPO=$PSCRATCH/BIBgen CONFIG=equivariant_denoiser.json TAG=v12 SCHEDULE=noise_schedule.csv \
#   sbatch --account=mxyz --job-name=generate_v12 --mail-user=me@princeton.edu --mail-type=END,FAIL \
#   generate_like_perlmutter.slurm
#
# Defaults to the `shared` QOS (bills per-GPU, not per-node) with a single GPU and 1/4 of a
# node's cores, which is plenty for generation. To run on a full node instead, add
#   --qos=regular --gpus-per-task=1
# and raise -c to 32 in the srun line below. Always keep -C gpu (Perlmutter has no separate GPU
# partition or Della-style --constraint=gpu40/mig).

#SBATCH --job-name=generate
#SBATCH --output=logs/%x_%j.out
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=06:00:00

set -euo pipefail

# Inside a batch job the script runs from a spool copy, so the repo cannot be found
# relative to $0. It comes from REPO= in the experiments file, via the submit wrapper.
: "${REPO:?REPO (path to the BIBgen checkout to run) is not set}"
MODEL_DIR=${MODEL_DIR:-$REPO/examples/training}
OUT_DIR=${OUT_DIR:-$REPO/examples/generation}
SIZES=${SIZES:-$REPO/examples/generation/test_sizes_large.csv}
GENERATE_ARGS=${GENERATE_ARGS:-}

: "${CONFIG:?CONFIG (model json in config/models/) is not set}"
: "${TAG:?TAG is not set}"
: "${SCHEDULE:?SCHEDULE (noise schedule csv in config/noise_schedules/, same as training) is not set}"

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
srun -n 1 -c 8 --cpu-bind=cores -G 1 --gpu-bind=none \
    python "$REPO/examples/generation/generate_like.py" "$model_path" \
    "$REPO/config/models/$CONFIG" "$REPO/config/noise_schedules/$SCHEDULE" "$SIZES" -t "$TAG" $GENERATE_ARGS

echo "=== finished at $(date)"