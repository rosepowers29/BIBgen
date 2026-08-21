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
``config/noise_schedule.csv`` exactly). ``cosine`` follows Nichol & Dhariwal's "Improved DDPM"
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

Model architecture is specified via a JSON config, e.g. ``config/equivariant_denoiser.json``:

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
``false``). See ``config/equivariant_denoiser_learned_variance.json`` for an example. ``train.py``
and ``generate_like.py`` handle the wiring automatically based on ``model.predict_variances`` — no
other CLI arguments change; ``noise_schedule`` is still required by both scripts since it also
fixes ``n_timesteps``.

## Training

```bash
python train.py [diffused_data] [noise_schedule] [model_config] -e [epochs] -b [batch_size] -t [tag]
```

``-t/--tag`` names the output files for this run (``denoiser_<tag>.pth``, ``history_<tag>.csv``);
defaults to the config filename stem if omitted. ``-o/--out`` overrides the output path directly.

``submit_train.sub`` reads ``(data, config, tag, schedule)`` rows from ``experiments.txt`` and
queues one condor job per row, so multiple runs (including runs against different noise
schedules) can be submitted at once without output files clobbering each other. ``schedule`` is
a filename within ``config/`` (e.g. ``noise_schedule.csv`` or ``noise_schedule_cosine.csv``) —
the whole ``config/`` directory is already transferred to the job, so no other change is needed
to use a new schedule.

## Generation

```bash
python generate_like.py [model.pth] [model_config] [noise_schedule] [size_file] -t [tag]
```

Produces ``<tag>_like.hdf5``. Use the same ``-t`` you trained with. ``submit_generate_like.sub``
fans this out the same way, reading ``(config, tag, schedule)`` rows from
``generate_experiments.txt``. ``schedule`` must be the same one the model with that ``tag`` was
trained with.

## Analysis

```bash
python plot_comparison.py [mc_file] [gen_file]
```

Output directory defaults to ``plots/<tag>``, inferred from the generated file's name. Whether
the energy feature needs exponentiating back from ``ln(E)`` is auto-detected from the MC file;
override with ``-l {auto,yes,no}`` if needed.

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
