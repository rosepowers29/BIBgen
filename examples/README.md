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
directly rather than through an additional Fourier encoding.

When ``use_position_encoding`` is ``true``, ``position_encoding_kind`` selects the mapping
(following Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low
Dimensional Domains"):

- ``"learned"`` (default) — the original per-axis encoding, with frequencies trained by gradient
  descent alongside the rest of the network (``FourierEncoding`` in ``models/common.py``, one
  independent encoder each for φ, s, z). Kept as a baseline; the paper (Appendix A.3) finds
  gradient descent does not meaningfully move — or improve on — well-scaled fixed frequencies.
- ``"positional"`` — fixed, deterministic, log-linearly spaced frequencies per axis
  (``PositionalEncoding``), axis-aligned by construction (the paper's "positional encoding").
- ``"gaussian"`` — fixed random frequencies sampled once from a Gaussian and shared across a
  *single joint* encoder over (φ, s, z) (``GaussianFourierFeatures``), rather than three
  independent per-axis encoders. This mixes coordinates the way the paper's random Fourier
  feature mapping does, and the paper finds it outperforms axis-aligned positional encoding,
  especially off-axis (Appendix A.5).

``"positional"`` and ``"gaussian"`` both require ``position_encoding_scale``: either a single
float shared across φ, s, z, or a ``[sigma_phi, sigma_s, sigma_z]`` triple. **This codebase starts
with independent per-axis scales** rather than one shared scale, because φ is restricted to a
narrow window (``--phi-window``, e.g. π/4) while s and z span the full detector — the "densely
clustered" φ coordinate likely needs a different frequency scale than s/z. If the φ window is
later widened toward the full detector extent, φ's frequency content may end up comparable to s/z,
at which point collapsing back to one shared scale is worth revisiting.
The best scale(s) are tuned per dataset on held-out validation loss; see
``training/make_sweep_configs.py`` for generating a scale-sweep grid of configs.

``"gaussian"`` additionally accepts ``position_encoding_seed`` (int, optional): seeds the random
draw of the frequency matrix ``B`` (see ``GaussianFourierFeatures`` in ``models/common.py``) so
it's reproducible across runs. Omitted by default, in which case every run draws an independent
random ``B`` — fine when scale differences are large and clear, but when a sweep turns up two
scales whose validation loss is nearly tied, that tie may just reflect which random ``B`` each one
happened to draw rather than a real difference. Rerunning the tied candidates with a few different
seeds each (see ``--seed`` on ``make_sweep_configs.py`` below) checks whether the ranking actually
holds up. Ignored for ``"learned"``/``"positional"``, which have no randomness to seed.

``predict_variances`` defaults to ``false``. When ``true``, the model predicts its own per-hit,
per-feature variance alongside the mean, turning the loss into a genuine Gaussian NLL rather than
a ``β_τ``-weighted MSE (the fixed noise-schedule value is used as the variance when this is
``false``). See ``config/equivariant_denoiser_learned_variance.json`` for an example. ``train.py``
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

``--loss`` selects the training objective:

- ``simple`` (default): direct NLL regression of the model's prediction against the one observed
  ``x_τ`` from that event's specific forward-diffusion sample, using ``β_τ`` (or the model's own
  learned variance, if ``predict_variances``) as the Gaussian's variance.
- ``nelbo``: the actual negative ELBO (Ho et al. 2020) — for ``τ ≥ 1``, KL-divergence between the
  model's predicted reverse distribution and the *true* forward-process posterior
  ``q(x_τ|x_{τ+1},x_0)`` (a closed form depending on the clean sample ``x_0``, not just the one
  noisy draw); at ``τ = 0`` this degenerates to the same reconstruction term ``simple`` already
  uses. See ``NELBOLoss`` in ``losses.py`` for the full derivation. Applies the same
  ``--variance-loss-weight`` decoupling as ``simple`` when ``predict_variances`` is ``true``.

``-t/--tag`` names the output files for this run (``denoiser_<tag>.pth``, ``history_<tag>.csv``);
defaults to the config filename stem if omitted. ``-o/--out`` overrides the output path directly.

``submit_train.sub`` reads ``(data, config, tag, schedule, loss)`` rows from ``experiments.txt``
and queues one condor job per row, so multiple runs (including runs against different noise
schedules or loss functions) can be submitted at once without output files clobbering each
other. ``loss`` is passed straight through as ``--loss`` (``simple`` or ``nelbo``); every row
must specify it explicitly now that the field exists. ``schedule`` is a filename within
``config/`` (e.g. ``noise_schedule.csv`` or ``noise_schedule_cosine.csv``) —
the whole ``config/`` directory is already transferred to the job, so no other change is needed
to use a new schedule.

### Fourier feature scale sweep

``training/make_sweep_configs.py`` generates a grid of ``position_encoding_scale`` configs for
the ``"positional"``/``"gaussian"`` encoders (see Model Configuration above), and appends matching
rows to a manifest in the same format ``experiments.txt`` uses, so the grid submits through the
existing ``submit_train.sub``/Condor workflow with no other changes:

```bash
python make_sweep_configs.py gaussian --phi 0.5 1 2 4 8 16 32 64 --s 1 --z 1 \
    --data diffused_cyl_phipi4_large_logE.hdf5 --schedule noise_schedule.csv \
    --manifest experiments_fourier_sweep.txt
```

For ``gaussian``, ``--seed`` reruns each ``(phi, s, z)`` triple once per given seed (via
``position_encoding_seed``) instead of a single uncontrolled draw — use this to check whether a
close/tied result from a scale sweep actually holds up across different random frequency draws,
e.g. rerunning just the tied candidates:

```bash
python make_sweep_configs.py gaussian --phi 4 --s 2 4 --z 1 --seed 0 1 2 \
    --data diffused_cyl_phipi4_large_logE.hdf5 --schedule noise_schedule.csv \
    --manifest experiments_fourier_sweep_phaseB_reseed.txt
```

Because a full 3-axis grid is combinatorially expensive, sweep one axis at a time (holding the
other two fixed at a reasonable default), picking the best scale by validation loss before moving
to the next axis — φ first (since it's the axis whose "densely clustered" window most likely
needs a different scale than s/z), then s, then z — and only train the combined best-per-axis
config to full length once all three are chosen.

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
