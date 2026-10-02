# Sourced by training/submit_train.sh and generation/submit_generate_like.sh.
#
# read_settings FILE KEY...
#   Reads `KEY=value` lines (e.g. `REPO=/scratch/gpfs/GROUP/me/BIBgen`) from an experiments
#   file into shell variables of the same name. Only the listed keys are accepted; anything
#   else of that shape is an error, so a typo cannot be silently ignored. `~` at the start of
#   a value expands to $HOME. Trailing `# comments` are allowed. Experiment rows (no `=`) are
#   left for the caller.
read_settings() {
    local file=$1 line key value
    shift
    while IFS= read -r line || [[ -n $line ]]; do
        line=${line%%#*}
        [[ $line =~ ^[[:space:]]*([A-Za-z_]+)[[:space:]]*=[[:space:]]*([^[:space:]]*)[[:space:]]*$ ]] || continue
        key=${BASH_REMATCH[1]}
        value=${BASH_REMATCH[2]/#\~/$HOME}
        if [[ " $* " != *" $key "* ]]; then
            echo "$file: unknown setting $key (allowed: $*)" >&2
            return 1
        fi
        printf -v "$key" '%s' "$value"
    done < "$file"
}

# is_setting WORD: true for the first word of a `KEY=value` line, so row loops can skip it.
is_setting() {
    [[ $1 =~ ^[A-Za-z_]+= || $2 == =* ]]
}
