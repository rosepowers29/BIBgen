# Preprocessing and Readying Data

## File Conversion
Run ``python convert_slcio_files.py -i [inputfiles] -o [outputfile]`` on a machine with pyLCIO configured. ``inputfiles`` should be LCIO format, ``outputfile`` should end with ``.hdf5``.

## Make Training Data

```bash
python make_training_data.py [mm_file] [mp_file] -o [outfile] -s [train,val,test] -c -p [phi_window]
```

By default, the energy feature is stored as ``ln(E)`` rather than raw ``E``. Pass ``--raw-energy``
to store raw ``E`` instead. The chosen setting is stamped into the output file
(``f["transformation"].attrs["log_energy"]``) so downstream scripts can detect it automatically.

## Make Diffused Data
Run ``diffuse.py [raw_data] [noise_schedule] -o [outfile]``.

## Noise Schedule

``noise_schedule.csv`` files are generated with ``make_noise_schedule.py``:

```bash
python make_noise_schedule.py quadratic [out.csv] -T [n_timesteps] --scale [scale]
python make_noise_schedule.py cosine [out.csv] -T [n_timesteps] --target-alpha-bar-t [alpha_bar_T]
```

``quadratic`` reproduces the original ``beta(tau) = scale * tau^2`` schedule (defaults match
``config/noise_schedules/noise_schedule.csv`` exactly). ``cosine`` follows Nichol & Dhariwal's "Improved DDPM"
schedule, solved so that the cumulative ``alpha_bar`` at the final timestep matches
``--target-alpha-bar-t`` exactly. Compared to the quadratic schedule, cosine spreads noise more
evenly through the middle of the chain, but concentrates a sharp jump in the final step or two;
raise ``--target-alpha-bar-t`` to soften that jump (``--s`` has little effect on it).

Compare schedules before committing to a full diffuse+train run with:

```bash
python plot_noise_schedule.py [schedule1.csv] [schedule2.csv] ... --labels [name1] [name2] ... -o [comparison.png]
```

This prints ``beta_min``/``beta_max``/``alpha_bar_T`` for each schedule and, if ``-o`` is given,
saves a plot of ``beta(tau)`` and ``alpha_bar(tau)``.

## Model Configuration

Model architecture is specified via a JSON config, e.g. ``config/models/equivariant_denoiser.json``:

```json
{
  "name": "EquivariantDenoiser",
  "hyperparameters": {
    "tau_encoding_dimension": 32,
    "position_encoding_dimension": 64,
    "hidden_layer_size": 256,
    "n_hidden_layers": 4,
    "use_position_encoding": false
  }
}
```

``use_position_encoding`` defaults to ``false`` — hit coordinates (φ, s, z) are fed to the model
directly rather than through an additional learned Fourier encoding.

``predict_variances`` defaults to ``false``. When ``true``, the model predicts its own per-hit,
per-feature variance alongside the mean, turning the loss into a genuine Gaussian NLL rather than
a ``β_τ``-weighted MSE (the fixed noise-schedule value is used as the variance when this is
``false``). See ``config/models/equivariant_denoiser_learned_variance.json`` for an example. ``train.py``
and ``generate_like.py`` handle the wiring automatically based on ``model.predict_variances`` — no
other CLI arguments change; ``noise_schedule`` is still required by both scripts since it also
fixes ``n_timesteps``.

When ``predict_variances`` is ``true``, training automatically uses ``DecoupledGaussianNLLLoss``
(mean and variance share every hidden layer, only diverging at the final output slice — without
decoupling, the variance head tends to collapse onto a trivial solution that just reproduces
``β_τ`` instead of learning anything input-dependent). ``--variance-loss-weight`` (default ``1.0``)
weights the variance-training term relative to the mean-training term.

## Training

```bash
python train.py [diffused_data] [noise_schedule] [model_config] -e [epochs] -b [batch_size] -t [tag]
```

``-t/--tag`` names the output files for this run (``denoiser_<tag>.pth``, ``history_<tag>.csv``);
defaults to the config filename stem if omitted. ``-o/--out`` overrides the output path directly.

``submit_train.sub`` reads ``(data, config, tag, schedule)`` rows from ``experiments.txt`` and
queues one condor job per row, so multiple runs (including runs against different noise
schedules) can be submitted at once without output files clobbering each other. ``schedule`` is
a filename within ``config/noise_schedules/`` (e.g. ``noise_schedule.csv`` or ``noise_schedule_cosine.csv``) —
the whole ``config/`` directory is already transferred to the job, so no other change is needed
to use a new schedule. Likewise ``config`` is a filename within ``config/models/``.

On Della (Slurm), ``./submit_train.sh experiments.txt`` is the equivalent: one ``train.slurm`` job
per row. Run it as a program (``./submit_train.sh`` or ``bash submit_train.sh``), never with
``source``, which would close your shell when the script exits. ``*.txt`` is gitignored, so each
person keeps their own experiments file. A complete one looks like this:

```
REPO=/scratch/gpfs/<GROUP>/<netid>/BIBgen   # required: your checkout; outputs and logs go here
MAIL_USER=<netid>@princeton.edu             # optional: email when each job ends or fails
RUNTIME=12:00:00                            # optional: wall-time limit (default 12:00:00)
DATA_DIR=/scratch/gpfs/IOJALVO/mz6011/BIBgen/data   # optional: default $REPO/data

# data                                config                      tag       schedule
# schedule must be the one `data` was diffused with
diffused_cyl_phipi4_large_logE.hdf5   equivariant_denoiser.json   v1        noise_schedule.csv
diffused_cyl_phipi4_large_logE.hdf5   mlp_denoiser.json           mlp_v1    noise_schedule.csv
```

Settings are ``KEY=value`` lines; anything else that is not a comment is a row, in the same
``data config tag schedule`` order as the Condor files (commas also work). A bare ``data`` name is
looked up in ``DATA_DIR``; a path is used as given. ``config`` and ``schedule`` are bare filenames in
``config/models/`` and ``config/noise_schedules/``. Pointing ``DATA_DIR`` at the shared copy above
saves duplicating the 26 GB file. ``diffused_cyl_phipi4_large_logE.hdf5`` was diffused with
``noise_schedule.csv``.

```bash
./submit_train.sh -n experiments.txt                     # print the sbatch commands only
./submit_train.sh experiments.txt --test-only            # ask the scheduler, create no jobs
./submit_train.sh experiments.txt                        # submit
EPOCHS=51 ./submit_train.sh experiments.txt --time=04:00:00   # env knobs; extra sbatch options last
squeue -u $USER --start                                  # estimated start times
```

The defaults in ``train.slurm`` (1 GPU on a 40 GB A100 or a 40 GB MIG slice, 2 cores, 4 GB, 12 h)
are sized from a measured run: 151 epochs of ``equivariant_denoiser.json`` on the file above took
8 h 57 min and used 1.3 cores, 1.6 GB of CPU memory and 6.3 GB of GPU memory. Scale ``RUNTIME`` with
epochs, model size and dataset; a shorter limit never queues longer and sometimes much shorter.
A tag whose ``denoiser_<tag>.pth`` already exists is skipped unless you pass ``-f``.

On Perlmutter (NERSC, also Slurm), ``./submit_train_perlmutter.sh experiments.txt`` is the
equivalent: one ``train_perlmutter.slurm`` job per row. It takes the same ``data config tag
schedule`` rows as above, plus one extra required setting:

```
REPO=$PSCRATCH/BIBgen                       # required: your checkout; outputs and logs go here
ACCOUNT=mxyz                                # required: NERSC charge account (sbatch -A; see `iris -a`)
MAIL_USER=<user>@princeton.edu              # optional: email when each job ends or fails
RUNTIME=12:00:00                            # optional: wall-time limit (default 12:00:00)
DATA_DIR=/path/to/shared/data               # optional: default $REPO/data
```

``train_perlmutter.slurm`` defaults to the ``shared`` QOS, which bills per-GPU rather than
per-node (each Perlmutter GPU node has 4 A100s) -- the right choice for a single-GPU job like
this one. Pass ``--qos=regular --nodes=N --ntasks=<N*4>`` (and edit the ``srun`` line) to spread
a run across full nodes instead. Usage otherwise matches ``submit_train.sh``:

```bash
./submit_train_perlmutter.sh -n experiments.txt
./submit_train_perlmutter.sh experiments.txt
```

## Hyperparameter Optimization

```bash
python hyperoptimize.py [diffused_data] [noise_schedule] [model_template] -e [epochs] -b [batch_size] -n [trials] --timeout [hours] -o [best.json]
```

``model_template`` is a model config in ``config/model_templates/`` in which a hyperparameter may be a
search range instead of a value, e.g. ``{"type": "int", "exponentiate": true, "range": [7, 9]}``
searches ``2**7`` to ``2**9``. Each optuna trial trains for ``-e`` epochs and is scored by its best
validation loss.

The study is saved as it runs to a ``.db`` next to ``-o`` with the same name (``best.json`` ->
``best.db``), and running the same command again resumes it. ``-n`` (default 40) is the total number
of trials over all runs: completed, pruned and errored (e.g. out of memory) trials count towards it,
while a trial cut off because the process was killed is run again. ``--timeout`` (default 20 hours) stops
new trials from starting after that time; the trial in progress finishes. The best config, in the format
``train.py`` reads, is written to ``-o`` only once all trials are done.

``submit_hyperoptimize.sub`` reads ``(data, config, out, schedule)`` rows from ``experiments_hyper.txt``,
where ``config`` is a template in ``config/model_templates/``. Each row's study lives in a folder named
after ``out``, e.g. ``best_equivariant_denoiser/`` holding ``best_equivariant_denoiser.db`` and, once
finished, ``best_equivariant_denoiser.json``. The folder is transferred to the job and back, so
``condor_submit submit_hyperoptimize.sub`` resumes each unfinished study where the last job left off;
resubmit until the ``.json`` appears. Delete the folder to start a study over.

## Generation

```bash
python generate_like.py [model.pth] [model_config] [noise_schedule] [size_file] -t [tag]
```

Produces ``<tag>_like.hdf5``. Use the same ``-t`` you trained with. ``submit_generate_like.sub``
fans this out the same way, reading ``(config, tag, schedule)`` rows from
``generate_experiments.txt``. ``schedule`` must be the same one the model with that ``tag`` was
trained with.

On Della, ``./submit_generate_like.sh generate_experiments.txt`` does the same on a MIG GPU. Its
file takes the same ``REPO=`` (required), ``MAIL_USER=`` and ``RUNTIME=`` (default ``06:00:00``)
settings, followed by ``config tag schedule`` rows:

```
REPO=/scratch/gpfs/<GROUP>/<netid>/BIBgen
MAIL_USER=<netid>@princeton.edu

# config                      tag       schedule (same as training)
equivariant_denoiser.json     v1        noise_schedule.csv
mlp_denoiser.json             mlp_v1    noise_schedule.csv
```

It reads ``$REPO/examples/training/denoiser_<tag>.pth`` and needs ``test_sizes_large.csv`` in
``$REPO/examples/generation/`` (from ``write_test_sizes.py``; the diffused file has no ``test``
group, so it must come from the matching raw file). To queue generation behind a training job:

```bash
./submit_generate_like.sh generate_experiments.txt --dependency=afterok:<train_jobid>
```

On Perlmutter, ``./submit_generate_like_perlmutter.sh generate_experiments.txt`` is the
equivalent, submitting ``generate_like_perlmutter.slurm`` jobs on the ``shared`` QOS. It takes the
same ``REPO=`` (required), ``ACCOUNT=`` (required, NERSC charge account), ``MAIL_USER=`` and
``RUNTIME=`` (default ``06:00:00``) settings and ``config tag schedule`` rows as the Della version
above.

### Arbitrary numbers of events

``generate_like.py`` reproduces the test-event sizes one for one, which suits a like-for-like
comparison but caps the output at the size of the test set. To generate any number of events,
first write a small generator cache from the raw training file, once:

```bash
python write_generator_cache.py [raw_training_file.hdf5] -o generator_cache.npz
```

The cache holds the sorted hit counts of every ``training`` and ``validation`` event (read from
dataset shapes only, so no hit data is loaded), the sphering ``mu``/``std``, and the ``log_energy``
flag. Files that predate the ``log_energy`` attribute, such as ``raw_cyl_phipi4_large.hdf5``, have
it inferred from the sign of the energy mean (``ln(E)`` of sub-GeV hits is negative), with a
warning. Prefer the raw file to the diffused one; the cache is the same and the raw file is far
smaller. Generation jobs then need only the ``.npz``, not the training data.

**The cache must come from the raw file the model's training data was diffused from** (e.g.
``raw_cyl_phipi4_large_logE.hdf5`` for a model trained on ``diffused_cyl_phipi4_large_logE.hdf5``).
The model outputs sphered values, and the cache's ``mu``/``std`` and ``log_energy`` turn them into
physical units, so a cache from a different parameterisation silently produces wrong energies.
Nothing in a ``.pth`` records what it was trained on, so this cannot be checked automatically.
Keep one cache per training file and name it after that file.

```bash
python from_cache.py [model.pth] [model_config] [noise_schedule] [generator_cache.npz] -o [tag]_gen.hdf5 -n [nevents]
```

Each event's size is drawn from the cached hit-count distribution (by interpolating its empirical
quantile function, so sizes stay within the training range). As with ``generate_like.py``,
``noise_schedule`` must be the one the model was trained with. ``-n`` defaults to 1000; expect
roughly 100 events a minute on one GPU. Events are generated and written one at a time, so memory
does not grow with ``-n``.

The output differs from ``<tag>_like.hdf5`` in two ways:

- **It is in physical units, not sphered.** Each event is ``(E [GeV], phi, s [mm], z [mm])``,
  with the sphering undone and ``ln(E)`` exponentiated when the cache's ``log_energy`` is set.
  The file is stamped with ``sphered=False`` and ``log_energy=False`` attributes, which
  ``plot_comparison.py`` reads so it does not transform the events a second time.
- **Event IDs are ``evt_0 ... evt_<n-1>``** and do not correspond to test events.

``submit_from_cache.sub`` (OSPool) and ``./submit_from_cache_perlmutter.sh generate_experiments.txt``
(Perlmutter) submit it exactly like their ``generate_like`` counterparts, from the same
``config tag schedule`` experiments file and with the same settings, options and dependency
handling. Each row writes ``<tag>_gen.hdf5``. Instead of a size file they take a cache and an
event count per job.

On OSPool these are macros at the top of ``submit_from_cache.sub``: ``cache = generator_cache.npz``
(any path; Condor transfers the file and the job uses its file name) and ``nevents = 1000``.
Override them on the command line:

```bash
condor_submit -append "nevents = 5000" submit_from_cache.sub
```

On Perlmutter they are two more optional settings in the experiments file, alongside ``REPO=`` and
``ACCOUNT=``. They apply to every row, and are read only from the file (not from the environment):

```
REPO=/pscratch/sd/<u>/<username>/BIBgen
ACCOUNT=<nersc_account>
CACHE=generator_cache.npz     # default; relative paths are relative to $REPO/examples/generation
NEVENTS=5000                  # events per job, default 1000

# config                      tag       schedule (same as training)
equivariant_denoiser.json     v13       noise_schedule.csv
```

There is one job per experiments row. To get more events for a tag, raise ``nevents`` rather than
listing the tag twice: both jobs would write ``<tag>_gen.hdf5`` with the same event IDs.

## Analysis

```bash
python plot_comparison.py [mc_file] [gen_file.hdf5],[name] ...
```

Each generated file is given as a ``path,name`` pair; ``name`` is the legend label used for that
dataset in the overlay, 2D and clustering plots. Pass more than one pair to compare several models
against the same MC on the same axes.

Output directory defaults to ``plots/<tag>``, inferred from the first generated file's name;
``-t/--tag`` overrides it. Whether the energy feature needs exponentiating back from ``ln(E)`` is
auto-detected from the MC file; override with ``-l {auto,yes,no}`` if needed. Generated files
from ``from_cache.py`` are already in physical units and say so in their file attributes; they are
loaded as they are, and ``-l`` applies only to the MC file and to ``generate_like.py`` output.
Both ``<tag>_like.hdf5`` and ``<tag>_gen.hdf5`` names give the tag ``<tag>``.

``wasserstein_distances.csv`` is written with one row per (generated dataset, variable). Its
``tag`` column is the experiment tag inferred from each generated file's name, so the rows still
join against ``history_<tag>.csv`` when several models are plotted in one run; ``-t/--tag`` sets
that column too when there is exactly one generated file.

## Ablation Analysis

Once you've run `plot_comparison.py` for each config/energy-parameterization variant (producing
a `wasserstein_distances.csv` per tag under `plots/<tag>/`), run:

```bash
python analyze_ablation.py --plots-dir plots --history-dir training/history -o plots/analysis
```

This reads every `wasserstein_distances.csv` found under `--plots-dir` (tag comes from the file's
own `tag` column — no hardcoded list to maintain) and every `history_<tag>.csv` found under
`--history-dir`, then writes four comparison plots to `-o`:

- ``wasserstein_by_variable.png`` — one bar chart per physical variable (energy, φ, η, s, z),
  comparing all configs
- ``wasserstein_grouped.png`` — all variables and configs together, log-scale
- ``pareto_aggregate_vs_valloss.png`` — normalized aggregate Wasserstein score vs. best validation
  loss. **Faceted by energy parameterization (raw-E vs. log-E) when both are present** — validation
  loss is not comparable across the two, since ``ln(E)`` and ``E`` correspond to different
  likelihood functions (a Jacobian term separates them). Wasserstein distance itself doesn't have
  this problem, since it's always computed on the untransformed physical variable.
- ``pareto_pairwise_all_combos.png`` — pairwise Pareto fronts across all five variables; dominance
  here is unaffected by axis scale, so no normalization is needed for this plot specifically (unlike
  the aggregate score above, which does normalize before summing).

Use ``--tag-order`` to fix the tag ordering/subset shown (comma-separated); otherwise tags are
auto-sorted from whatever's present in the data.
