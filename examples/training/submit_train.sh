#!/bin/bash
# Submits one train.slurm job per row of an experiments file -- the Slurm counterpart of
# `queue data,config,tag,schedule from experiments_sched.txt` in submit_train.sub.
#
#   ./submit_train.sh [-n|--dry-run] [-f|--force] EXPERIMENTS_FILE [extra sbatch options...]
#
# The file starts with settings, one `KEY=value` per line, which make it self-contained so
# that each collaborator's file points at their own copy of the repo:
#   REPO=/scratch/gpfs/GROUP/me/BIBgen   required: the checkout whose code and config/ are used,
#                                        and where outputs and logs are written
#   MAIL_USER=me@princeton.edu           optional: email when each job ends or fails
#   RUNTIME=48:00:00                     optional: wall-time limit per job, any sbatch --time
#                                        format (default 12:00:00, from train.slurm)
#   DATA_DIR=/path/to/data               optional: where bare data names are looked up
#                                        (default $REPO/data)
#
# Then each row is `data config tag schedule` (whitespace- or comma-separated, so existing condor
# experiments files work unchanged). `#` comments and blank lines are skipped.
#   data      diffused hdf5: a bare name is looked up in DATA_DIR, a path is used as given
#   config    model json in config/
#   tag       names denoiser_<tag>.pth and history_<tag>.csv (written to $REPO/examples/training/)
#   schedule  noise schedule csv in config/ -- must be the one `data` was diffused with
#
# Extra options are passed straight to sbatch and override train.slurm's #SBATCH defaults,
# e.g. --constraint=gpu80, and also override MAIL_USER's --mail-type=END,FAIL and RUNTIME.
# Environment variables read by train.slurm (EPOCHS, BATCH_SIZE, TRAIN_ARGS, OUT_DIR) are
# forwarded too.
# -n prints the sbatch commands without submitting. -f submits even if the tag's
# denoiser_<tag>.pth already exists (it will be overwritten).

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

REPO="" MAIL_USER="" RUNTIME="" DATA_DIR=""
read_settings "$experiments" REPO MAIL_USER RUNTIME DATA_DIR
if [[ -z $REPO ]]; then
    echo "$experiments: add a REPO=/path/to/your/BIBgen line (the checkout jobs should run from)" >&2
    exit 2
fi
REPO=$(realpath "$REPO")
if [[ ! -f $REPO/examples/training/train.slurm ]]; then
    echo "REPO=$REPO does not look like a BIBgen checkout (no examples/training/train.slurm)" >&2
    exit 2
fi
if [[ $REPO != "$(realpath "$here/../..")" ]]; then
    echo "Note: submitting into REPO=$REPO, not the checkout this script lives in." >&2
fi
DATA_DIR=${DATA_DIR:-$REPO/data}
OUT_DIR=${OUT_DIR:-$REPO/examples/training}
export REPO DATA_DIR OUT_DIR

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

cd "$REPO/examples/training"
mkdir -p logs

status=0
while read -r data config tag schedule extra; do
    [[ -z $data || $data == \#* ]] && continue
    is_setting "$data" "$config" && continue
    if [[ -z $schedule || -n $extra ]]; then
        echo "Skipping malformed row (want: data config tag schedule): $data $config $tag $schedule $extra" >&2
        status=1; continue
    fi

    [[ $data == */* ]] && data_path=$data || data_path=$DATA_DIR/$data
    missing=""
    [[ -f $data_path ]] || missing+=" $data_path"
    [[ -f $REPO/config/$config ]] || missing+=" config/$config"
    [[ -f $REPO/config/$schedule ]] || missing+=" config/$schedule"
    if [[ -n $missing ]]; then
        echo "Skipping $tag: missing$missing" >&2
        status=1; continue
    fi
    if [[ -e $OUT_DIR/denoiser_$tag.pth && $force -eq 0 ]]; then
        echo "Skipping $tag: $OUT_DIR/denoiser_$tag.pth exists (use -f to overwrite)" >&2
        status=1; continue
    fi

    cmd=(sbatch --job-name="train_$tag" --output="logs/train_${tag}_%j.out"
         --export="ALL,DATA=$data,CONFIG=$config,TAG=$tag,SCHEDULE=$schedule"
         "${mail_opts[@]}" "${time_opts[@]}" "$@" train.slurm)
    if [[ $dry_run -eq 1 ]]; then
        echo "${cmd[*]}"
    else
        echo -n "$tag: "
        "${cmd[@]}"
    fi
done < <(tr ',' ' ' < "$experiments")

exit $status
