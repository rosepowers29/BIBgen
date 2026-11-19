#!/bin/bash
# Slurm (Perlmutter) counterpart of submit_from_cache.sub (OSG): generates <tag>_gen.hdf5 from
# denoiser_<tag>.pth, with NEVENTS events whose sizes are drawn from the generator cache CACHE
# (see write_generator_cache.py). Events are written in physical units, not sphered.
#
# Normally submitted by submit_from_cache_perlmutter.sh, which passes REPO/ACCOUNT/CACHE/NEVENTS from the
# experiments file. To submit a single run by hand, from $REPO/examples/generation/:
#
#   mkdir -p logs
#   REPO=$PSCRATCH/BIBgen CONFIG=equivariant_denoiser.json TAG=v12 SCHEDULE=noise_schedule.csv \
#   NEVENTS=1000 sbatch --account=mxyz --job-name=from_cache_v12 --mail-user=me@princeton.edu \
#   --mail-type=END,FAIL from_cache_perlmutter.slurm
#
# Defaults to the `shared` QOS (bills per-GPU, not per-node) with a single GPU and 1/4 of a
# node's cores, which is plenty for generation. To run on a full node instead, add
#   --qos=regular --gpus-per-task=1
# and raise -c to 32 in the srun line below. Always keep -C gpu (Perlmutter has no separate GPU
# partition or Della-style --constraint=gpu40/mig).

#SBATCH --job-name=from_cache
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
CACHE=${CACHE:-$REPO/examples/generation/generator_cache.npz}
NEVENTS=${NEVENTS:-1000}
GENERATE_ARGS=${GENERATE_ARGS:-}

: "${CONFIG:?CONFIG (model json in config/models/) is not set}"
: "${TAG:?TAG is not set}"
: "${SCHEDULE:?SCHEDULE (noise schedule csv in config/noise_schedules/, same as training) is not set}"

model_path=$MODEL_DIR/denoiser_$TAG.pth
for f in "$model_path" "$CACHE"; do
    [[ -f $f ]] || { echo "Missing input: $f" >&2; exit 1; }
done

if [[ ! -x $REPO/.venv/bin/python ]]; then
    echo "No environment at $REPO/.venv -- run 'uv sync' on a login node first." >&2
    exit 1
fi
source "$REPO/.venv/bin/activate"
export PYTHONUNBUFFERED=1

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "=== tag=$TAG model=$model_path config=$CONFIG schedule=$SCHEDULE cache=$CACHE nevents=$NEVENTS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$OUT_DIR"
# shellcheck disable=SC2086  # GENERATE_ARGS is deliberately word-split
srun -n 1 -c 8 --cpu-bind=cores -G 1 --gpu-bind=none \
    python "$REPO/examples/generation/from_cache.py" "$model_path" \
    "$REPO/config/models/$CONFIG" "$REPO/config/noise_schedules/$SCHEDULE" "$CACHE" \
    -o "${TAG}_gen.hdf5" -n "$NEVENTS" $GENERATE_ARGS

echo "=== finished at $(date)"
