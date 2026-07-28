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

## Training

```bash
python train.py [diffused_data] [noise_schedule] [model_config] -e [epochs] -b [batch_size] -t [tag]
```

``-t/--tag`` names the output files for this run (``denoiser_<tag>.pth``, ``history_<tag>.csv``);
defaults to the config filename stem if omitted. ``-o/--out`` overrides the output path directly.

``submit_train.sub`` reads ``(data, config, tag)`` rows from ``experiments.txt`` and queues one
condor job per row, so multiple runs can be submitted at once without output files clobbering
each other.

## Generation

```bash
python generate_like.py [model.pth] [model_config] [noise_schedule] [size_file] -t [tag]
```

Produces ``<tag>_like.hdf5``. Use the same ``-t`` you trained with. ``submit_generate_like.sub``
fans this out the same way, reading ``(config, tag)`` pairs from ``generate_experiments.txt``.

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
