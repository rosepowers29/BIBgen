#!/bin/bash
# Submits one generate_like.slurm job per row of an experiments file -- the Slurm counterpart
# of `queue config,tag,schedule from generate_experiments.txt` in submit_generate_like.sub.
#
#   ./submit_generate_like.sh [-n|--dry-run] [-f|--force] EXPERIMENTS_FILE [extra sbatch options...]
#
# The file starts with settings, one `KEY=value` per line, which make it self-contained so
# that each collaborator's file points at their own copy of the repo:
#   REPO=/scratch/gpfs/GROUP/me/BIBgen   required: the checkout whose code, config/ and trained
#                                        models are used, and where outputs and logs are written
#   MAIL_USER=me@princeton.edu           optional: email when each job ends or fails
#   RUNTIME=12:00:00                     optional: wall-time limit per job, any sbatch --time
#                                        format (default 06:00:00, from generate_like.slurm)
#
# Then each row is `config tag schedule` (whitespace- or comma-separated, so existing condor
# experiments files work unchanged). `#` comments and blank lines are skipped.
#   config    model json in config/, as trained
#   tag       reads $REPO/examples/training/denoiser_<tag>.pth,
#             writes $REPO/examples/generation/<tag>_like.hdf5
#   schedule  noise schedule csv in config/ -- must be the one the model was trained with
#
# Extra options are passed straight to sbatch and override generate_like.slurm's #SBATCH
# defaults, including MAIL_USER's --mail-type=END,FAIL and RUNTIME's --time. Use --dependency=afterok:<jobid> to
# queue generation behind a training job that has not finished yet (a missing model is then
# only warned about here). Environment variables read by generate_like.slurm (SIZES,
# GENERATE_ARGS, MODEL_DIR, OUT_DIR) are forwarded too. -n prints the sbatch commands without submitting. -f submits even if
# <tag>_like.hdf5 already exists (it will be overwritten).

set -euo pipefail

dry_run=0
force=0
while [[ $# -gt 0 && $1 == -* ]]; do
    case $1 in
        -n|--dry-run) dry_run=1 ;;
        -f|--force) force=1 ;;
        *) echo "Unknown option $1 (sbatch options go after the experiments file)" >&2; exit 2 ;;
    esac
    shift
done
if [[ $# -lt 1 ]]; then
    sed -n '2,/^$/s/^# \{0,1\}//p' "$0"
    exit 2
fi
[[ -f $1 ]] || { echo "No such experiments file: $1" >&2; exit 2; }
experiments=$(realpath "$1")
shift

here=$(cd "$(dirname "$0")" && pwd)
source "$here/../slurm_settings.sh"

REPO="" MAIL_USER="" RUNTIME=""
read_settings "$experiments" REPO MAIL_USER RUNTIME
if [[ -z $REPO ]]; then
    echo "$experiments: add a REPO=/path/to/your/BIBgen line (the checkout jobs should run from)" >&2
    exit 2
fi
REPO=$(realpath "$REPO")
if [[ ! -f $REPO/examples/generation/generate_like.slurm ]]; then
    echo "REPO=$REPO does not look like a BIBgen checkout (no examples/generation/generate_like.slurm)" >&2
    exit 2
fi
if [[ $REPO != "$(realpath "$here/../..")" ]]; then
    echo "Note: submitting into REPO=$REPO, not the checkout this script lives in." >&2
fi
MODEL_DIR=${MODEL_DIR:-$REPO/examples/training}
OUT_DIR=${OUT_DIR:-$REPO/examples/generation}
SIZES=${SIZES:-$REPO/examples/generation/test_sizes_large.csv}
export REPO MODEL_DIR OUT_DIR SIZES

mail_opts=()
[[ -n $MAIL_USER ]] && mail_opts=(--mail-user="$MAIL_USER" --mail-type=END,FAIL)

# Slurm accepts MM, MM:SS, HH:MM:SS, D-HH, D-HH:MM and D-HH:MM:SS.
time_opts=()
if [[ -n $RUNTIME ]]; then
    if [[ ! $RUNTIME =~ ^([0-9]+-)?[0-9]+(:[0-9]+){0,2}$ ]]; then
        echo "$experiments: RUNTIME=$RUNTIME is not a Slurm time (e.g. 23:59:00 or 2-00:00:00)" >&2
        exit 2
    fi
    time_opts=(--time="$RUNTIME")
fi

if [[ ! -f $SIZES ]]; then
    echo "Missing size file $SIZES -- see write_test_sizes.py (needs the raw_*.hdf5 with a test group)." >&2
    exit 1
fi

cd "$REPO/examples/generation"
mkdir -p logs

has_dependency=0
for opt in "$@"; do [[ $opt == --dependency* || $opt == -d* ]] && has_dependency=1; done

status=0
while read -r config tag schedule extra; do
    [[ -z $config || $config == \#* ]] && continue
    is_setting "$config" "$tag" && continue
    if [[ -z $schedule || -n $extra ]]; then
        echo "Skipping malformed row (want: config tag schedule): $config $tag $schedule $extra" >&2
        status=1; continue
    fi

    missing=""
    [[ -f $REPO/config/$config ]] || missing+=" config/$config"
    [[ -f $REPO/config/$schedule ]] || missing+=" config/$schedule"
    if [[ -n $missing ]]; then
        echo "Skipping $tag: missing$missing" >&2
        status=1; continue
    fi
    if [[ ! -f $MODEL_DIR/denoiser_$tag.pth ]]; then
        if [[ $has_dependency -eq 0 ]]; then
            echo "Skipping $tag: missing $MODEL_DIR/denoiser_$tag.pth" >&2
            status=1; continue
        fi
        echo "Note: $tag has no denoiser yet; relying on the --dependency to produce it." >&2
    fi
    if [[ -e $OUT_DIR/${tag}_like.hdf5 && $force -eq 0 ]]; then
        echo "Skipping $tag: $OUT_DIR/${tag}_like.hdf5 exists (use -f to overwrite)" >&2
        status=1; continue
    fi

    cmd=(sbatch --job-name="generate_$tag" --output="logs/generate_${tag}_%j.out"
         --export="ALL,CONFIG=$config,TAG=$tag,SCHEDULE=$schedule"
         "${mail_opts[@]}" "${time_opts[@]}" "$@" generate_like.slurm)
    if [[ $dry_run -eq 1 ]]; then
        echo "${cmd[*]}"
    else
        echo -n "$tag: "
        "${cmd[@]}"
    fi
done < <(tr ',' ' ' < "$experiments")

exit $status
