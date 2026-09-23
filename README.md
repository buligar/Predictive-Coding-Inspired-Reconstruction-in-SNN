# Predictive-Coding-Inspired Reconstruction of Nonlinear Dynamics in Spiking Neural Networks

Python code, numerical results, and figures accompanying the manuscript by **Bulat B. Batuev and Sergey V. Sukhov**:

[**Predictive-Coding-Inspired Reconstruction of Nonlinear Dynamics in Spiking Neural Networks** — revised manuscript, September 14, 2026].

The study compares two prediction-error architectures for reconstructing an autonomous sensory state in a spiking neural network. Both use leaky integrate-and-fire (LIF) populations, the Neural Engineering Framework (NEF), and a local Prescribed Error Sensitivity (PES)-like decoder update. Experiments cover a periodic two-dimensional oscillator and the chaotic Lorenz system.

The revised manuscript examines reconstruction quality, runtime, emitted spike rate, hyperparameter sensitivity, latent-state clipping, and reconstruction after freezing decoder adaptation.

![Lorenz reconstruction: reference, autonomous sensory state, and top-down prediction](figs/3_3_new.png)

## Model and terminology

| Architecture | Meaning | Prediction-error pathway |
| --- | --- | --- |
| **PC-EC** | Predictive-coding architecture with an **external comparator** | The continuous error directly drives the latent state. |
| **PC-SC** | Predictive-coding architecture with a **spiking comparator** | A dedicated LIF population encodes the error; its decoded activity drives the latent state. |

The sensory population `o1` generates the state to be reconstructed. Its recurrent decoder is fitted offline to the reference dynamics. During the initial cue interval, the reference initializes and stabilizes the sensory state; after cue removal, the sensory population evolves autonomously. The latent population `o2` represents an internal state `z`, and an adaptive top-down decoder produces the reconstruction `g`.

```text
PC-EC:  sensory state o1 ──► e = o1 − g ───────────────────► latent z ──► g
                               ▲                                       │
                               └───────────────────────────────────────┘

PC-SC:  sensory state o1 ──► e = o1 − g ──► error LIF population ──► latent z ──► g
                               ▲                                               │
                               └───────────────────────────────────────────────┘
```

In both implementations, the top-down decoder is updated from filtered latent spikes and the **continuous** prediction error:

```python
W_pred += eta * np.outer(a_z, e) * dt
```

PC-SC adds a spiking representation to the bottom-up state-update pathway; the implementation still uses continuous decoded states and a continuous error in the learning rule. The latent state is an internal representation and need not reproduce the sensory trajectory point by point.

The manuscript reports similar reconstruction fidelity for the two architectures near the baseline settings. PC-SC has higher runtime and spiking activity, while remaining stable over a wider tested learning-rate range. These are CPU simulation results, not measurements of neuromorphic hardware energy consumption.

## Repository guide

| File or directory | Purpose |
| --- | --- |
| [PC-EC_PC-SC_lorenz.py](PC-EC_PC-SC_lorenz.py) | Representative Lorenz runs for both architectures; default `N_sens = 540`. Saves trajectories as plots, spike rasters, and summary metrics. |
| [PC-EC_PC-SC_osc.py](PC-EC_PC-SC_osc.py) | Corresponding oscillator runs; default `N_sens = 20`. |
| [compare_bio_nebio_full_new.py](compare_bio_nebio_full_new.py) | Main repeated comparison used for the revised eight-metric protocol, including emitted spikes per simulated second. |
| [compare_graph_new.py](compare_graph_new.py) | Builds publication-style 4 × 2 comparison plots from the saved spike-rate results workbook. |
| [ablation_lorenz.py](ablation_lorenz.py) | Eight-parameter sensitivity analysis for Lorenz at `N_sens = 540`, with ten seeds. |
| [ablation_osc.py](ablation_osc.py) | Corresponding sensitivity analysis for the oscillator at `N_sens = 20`. |
| [clipping_cd_8panels_spacing.py](clipping_cd_8panels_spacing.py) | Supplementary 4 × 2 view of prediction and latent trajectories with clipping disabled/enabled, for both systems. |
| [PC-EC_PC-SC_lorenz_new.py](PC-EC_PC-SC_lorenz_new.py) | Frozen-decoder experiment: PES adaptation stops at 15 s. |
| [compare_bio_nebio_full.py](compare_bio_nebio_full.py), [compare_graph.py](compare_graph.py) | Earlier comparison and plotting workflow, including Python peak memory instead of spike rate. |
| [results_PC-SC_PC-EC_10tests_8metrics_4x2_spikes/](results_PC-SC_PC-EC_10tests_8metrics_4x2_spikes/) | Revised repeated-run results: configuration, benchmark environment, raw/aggregated CSV and XLSX tables, and figures. |
| [results_PC-SC_PC-EC_10tests_8metrics_4x2/](results_PC-SC_PC-EC_10tests_8metrics_4x2/) | Earlier repeated-run results with memory measurements. |
| [results_PC-SC_nePC-SC_lorenz_oscillator/](results_PC-SC_nePC-SC_lorenz_oscillator/) | Representative-run, clipping, and frozen-decoder figures, plus the most recently saved single-run tables. |
| [results_ablation/](results_ablation/), [results_ablation_quicktest/](results_ablation_quicktest/) | Saved sensitivity-analysis tables and figures. |
| [ijbc_figures/](ijbc_figures/), [figs/](figs/) | Exported comparison figures and selected illustrations. |

The historical `bio`, `nebio`, `nePC-SC`, and `ijbc` filenames are retained for compatibility. Use the PC-EC/PC-SC definitions above when interpreting the experiments.

## Installation

The simulations are implemented directly in **NumPy** and run on the CPU. Nengo and a GPU are not required.

The saved benchmark environment records Python **3.10.12**, NumPy **1.26.4**, pandas **2.2.3**, and Matplotlib **3.10.1**. To use the recorded numerical-library versions, create an environment with Python 3.10 and run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy==1.26.4 pandas==2.2.3 matplotlib==3.10.1 openpyxl psutil
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

`openpyxl` supports the Excel tables. `psutil` is optional and enables process-memory measurements in the representative-run and earlier comparison scripts; its absence does not prevent the simulations from running. Versions of these two packages were not recorded in the benchmark file.

Run all commands below from the repository root. For a headless Linux environment, use `export MPLBACKEND=Agg` before launching scripts.

## Running the experiments

Settings are Python dataclass fields and module constants; the scripts do not expose a command-line configuration interface. Full sweeps execute many simulations. Set a distinct `out_dir` for each experiment to preserve existing results: several scripts share output directories and overwrite identically named CSV and configuration files.

### 1. Representative reconstructions

```bash
python PC-EC_PC-SC_osc.py
python PC-EC_PC-SC_lorenz.py
```

Each script runs both architectures at one population size by default. Adjust `Config.signal_names`, `n_sens_values`, `T`, `base_seed`, and `out_dir` near the beginning of the file as needed.

Outputs include `config.txt`, `summary_metrics.csv`, `summary_metrics_partial.csv`, and `figures/overview_*.png`, `components_*.png`, and `comparison_*.png`. Overview figures contain the reference, autonomous sensory trajectory, prediction, latent state, prediction error, and spike rasters. An error-population raster exists only for PC-SC.

### 2. Repeated comparison: revised manuscript Figures 2 and 4

```bash
python compare_bio_nebio_full_new.py
```

The default protocol matches the sensory-neuron sweep in manuscript Table 1:

| Setting | Value |
| --- | --- |
| Systems | 2-D oscillator and 3-D Lorenz attractor |
| Sensory neurons | `10, 60, 110, …, 560` |
| Tests per system, population size, and architecture | 10 |
| Latent neurons | 50 |
| Error neurons | 50 in PC-SC; no error population in PC-EC |
| Simulation duration / time step | 30 s / 0.001 s |
| Cue duration | 5 s |
| Reconstruction-metric evaluation starts | 5.5 s |
| Synaptic / membrane / refractory time constants | 0.05 / 0.02 / 0.002 s |
| PES learning rate | `1e-3` |
| Sensory/error decoder ridge regularization | `1e-2` / `1e-2` |
| Oscillator angular frequency | `2π` rad/s (1 Hz) |
| Lorenz parameters | `sigma = 10`, `rho = 28`, `beta = 2.667` |

This produces **480 architecture runs**: two systems × twelve population sizes × ten tests × two architectures. Each test shares its sensory population and recurrent decoder between architectures.

For a smaller trial without editing the script, use its Python API:

```bash
python - <<'PY'
from dataclasses import replace
import compare_bio_nebio_full_new as experiment

cfg = replace(
    experiment.CFG,
    signal_names=("oscillator",),
    n_sens_values=(20,),
    n_tests=1,
    out_dir="results_trial_oscillator",
)
experiment.run_sweep(cfg)
PY
```

The main sweep saves the following in `results_PC-SC_PC-EC_10tests_8metrics_4x2_spikes/`:

- `summary_metrics_raw.csv` / `.xlsx`: one row per run;
- `summary_metrics_mean_std.csv` / `.xlsx`: means and standard deviations across tests;
- `config.txt` and `benchmark_environment.txt`;
- `figures/mean_std_<signal>_8metrics_4x2.png` / `.pdf`.

To regenerate publication-style plots from the included workbook without rerunning simulations:

```bash
python compare_graph_new.py
```

This exports PNG, PDF, and SVG figures to `ijbc_figures/`, together with `mean_std_used_for_ijbc_plot.xlsx`. If you use another results directory, change `XLSX_PATH`. The older `compare_graph.py` has a default path whose architecture-name order differs from the included directory; correct that path before using it.

### 3. Hyperparameter sensitivity: Figures 7 and 8

```bash
python ablation_osc.py
python ablation_lorenz.py
```

Both scripts already import the existing `PC-EC_PC-SC_lorenz.py` module through `importlib`; this module supports both reference systems. The oscillator script selects its system through `AblationConfig.signal_names`.

The scripts vary one parameter at a time: PES learning rate, synaptic time constant, membrane time constant, refractory period, sensory-decoder regularization, error-decoder regularization, cue duration, and latent clipping bound. The `tau_syn` sweep changes `tau_syn`, `tau_o1`, and `tau_z` together. For `z_clip`, zero disables clipping; positive values constrain each latent component to `[-z_clip, z_clip]`. This setting is separate from clipping the PC-SC error-population input to `[-1, 1]`.

Configure `ABLATION_GRID`, `AblationConfig`, and its `out_dir` before running. Outputs are `ablation_raw.csv`, `ablation_raw_partial.csv`, `ablation_aggregated.csv`, and `figures/ablation_rmse_grid_<signal>.png` / `.pdf`.

The checked-in grid is broader than manuscript Table 2: it includes `eta = 0.1`, ridge values down to `1e-5`, and a 0.1 s cue. Table 2 reports learning rates through `1e-2`, ridge values from `1e-3` to `1e-1`, and cue durations from 1 to 20 s. Restrict the grids to the reported ranges when comparing with that table.

`QUICK_TEST = True` selects a smaller grid and three seeds, but the current quick-test branch in **both files** explicitly uses Lorenz with 540 sensory neurons and a 30 s simulation. Edit that branch if an oscillator or shorter trial is intended.

### 4. Clipping illustrations

```bash
python clipping_cd_8panels_spacing.py
```

The default entry point runs PC-SC with `z_clip = 0` and `1` for the oscillator (`N_sens = 20`) and Lorenz (`N_sens = 540`). It reuses seeds within each system and saves prediction/latent-state panels to `figures/clipping_cd_8panels_PC_SC.png` under `Config.out_dir`. Change the entry-point `arch` argument to `"PC-EC"` for the external-comparator version. The quantitative ablation also includes `z_clip = 2`.

### 5. Frozen decoder: Figure 9

```bash
python PC-EC_PC-SC_lorenz_new.py
```

The default experiment uses Lorenz with 540 sensory neurons and runs both architectures:

| Interval | Sensory input and decoder adaptation |
| --- | --- |
| 0–5 s | Reference cue is applied; PES is active. |
| 5–15 s | Sensory dynamics are autonomous; PES remains active. |
| 15–30 s | Top-down decoder weights are frozen; error feedback remains active. |

The learning window is controlled by `Config.pes_start = 0.0` and `Config.pes_end = 15.0`. Overview and component plots mark cue removal and termination of learning.

The manuscript reports increased reconstruction error after freezing, while the prediction retains structured Lorenz-like dynamics. This is reconstruction with fixed learned weights **inside an active error-correcting loop**, not an isolated open-loop forecast. The script's default summary metrics span the post-cue evaluation window; they do not separately summarize the active-learning and frozen intervals.

## Metrics and interpretation

The revised repeated comparison saves eight metrics:

| CSV field | Definition |
| --- | --- |
| `sim_time_s` | Wall-clock time for online SNN simulation and PES adaptation. |
| `spikes_per_s` | Total emitted spikes from all simulated populations divided by simulated duration, including the cue interval; not a per-neuron firing rate. |
| `mean_amp_corr` | Mean component-wise Pearson correlation between analytic-signal amplitude envelopes of `o1` and `g`. |
| `mean_freq_corr` | Mean component-wise correlation between their instantaneous frequencies. |
| `mean_plv` | Mean component-wise phase-locking value. |
| `mean_phase_deg` | Component average of the absolute circular mean phase difference, in degrees. |
| `rmse_ref_o1` | RMSE between the reference and autonomous sensory state. |
| `rmse_o1_g` | RMSE between the sensory state and its top-down reconstruction. |

The analytic signal is computed with an FFT-based Hilbert-transform construction. Reconstruction metrics use the post-cue window; the default begins 0.5 s after cue removal. The two RMSE measures answer different questions: autonomous generation of the reference dynamics versus reconstruction of the sensory population. For chaotic Lorenz trajectories, long-term pointwise separation from the reference is expected and should not alone be interpreted as failure to reproduce the attractor dynamics.

For the revised benchmark, runtime excludes population initialization, offline sensory-decoder fitting, static PC-SC error-decoder fitting, and metric post-processing. Earlier scripts use different measurement procedures, so their runtime columns are not interchangeable with this benchmark.

## Saved data and reproducibility notes

The repeated-run CSV files and final representative-run summary use **semicolon delimiters and decimal commas**:

```python
import pandas as pd

raw = pd.read_csv(
    "results_PC-SC_PC-EC_10tests_8metrics_4x2_spikes/summary_metrics_raw.csv",
    sep=";",
    decimal=",",
)
```

Ablation CSV files and representative-run `summary_metrics_partial.csv` use standard comma delimiters and decimal points.

The included revised comparison contains 480 raw rows covering both systems. Other result directories combine figures from several runs: the current `results_ablation/ablation_raw.csv` contains only oscillator results, although figures for both systems are present; the representative-run summary currently contains two Lorenz rows. Use separate output directories when regenerating tables for each system and for the frozen-decoder experiment.

The representative-run scripts, including the frozen-decoder version used by Figure 9, filter predictions with a hard-coded `tau_g = 0.02` s. The repeated-comparison scripts use `cfg.tau_syn = 0.05` s for that filter. These workflows therefore do not have identical numerical settings despite sharing the architecture names.

The [saved benchmark environment](results_PC-SC_PC-EC_10tests_8metrics_4x2_spikes/benchmark_environment.txt) describes the original CPU experiment. When rerunning elsewhere, update the `benchmark_*` fields: the script reads Python/library versions automatically, but its hardware and operating-system descriptions are configured strings. Runtime depends on the local machine and numerical-library environment.

## Citation

Until publication details are finalized, cite the supplied manuscript as:

> Batuev, B. B., and Sukhov, S. V. (2026). *Predictive-Coding-Inspired Reconstruction of Nonlinear Dynamics in Spiking Neural Networks*. Revised manuscript, September 14, 2026.

Repository: [buligar/Predictive-Coding-Inspired-Reconstruction-in-SNN](https://github.com/buligar/Predictive-Coding-Inspired-Reconstruction-in-SNN).

## License

The repository code is distributed under the [GNU Affero General Public License v3.0](LICENSE). See the manuscript for its own publication and licensing information.
