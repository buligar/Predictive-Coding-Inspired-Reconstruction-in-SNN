"""
PC-SC vs PC-EC: 10 tests per condition + selected metrics + one 4x2 mean±std figure.

Saved and plotted metrics:
1) sim_time_s
2) spikes_per_s
3) mean_amp_corr
4) mean_freq_corr
5) mean_plv
6) mean_phase_deg
7) rmse_ref_o1
8) rmse_o1_g

Figure layout:
4 rows x 2 columns
Row 1: sim_time_s, spikes_per_s
Row 2: mean_amp_corr, mean_freq_corr
Row 3: mean_plv, mean_phase_deg
Row 4: rmse_ref_o1, rmse_o1_g

Important benchmarking convention:
- sim_time_s measures only the online SNN simulation and PES adaptation.
- offline decoder fitting and population initialization are performed before the timer starts.
- synchronization/RMSE metric post-processing is performed after the timer stops.
- spikes_per_s is the total number of spikes emitted by all spiking populations
  participating in the online architecture, divided by simulated duration.

Outputs:
- summary_metrics_raw.csv/xlsx      : all runs, selected columns only
- summary_metrics_mean_std.csv/xlsx : mean ± std by signal, N_sens, arch
- one large 4x2 figure per signal with mean ± std across 10 tests
- benchmark_environment.txt         : hardware/software environment used for the benchmark
"""

import gc
import time
import math
import platform
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    # Use ("lorenz",), ("oscillator",), or ("lorenz", "oscillator")
    signal_names: tuple = ("oscillator","lorenz")

    # Main sweep
    n_sens_values: tuple = tuple(range(10, 561, 50))

    # Independent tests per condition
    n_tests: int = 10

    # Common simulation
    T: float = 30.0
    dt: float = 0.001

    # Lorenz parameters
    rho: float = 28.0
    sigma: float = 10.0
    beta: float = 2.667
    x0_lorenz: tuple = (1.0, 1.0, 1.0)


    # Oscillator parameters
    oscillator_omega: float = 2.0 * math.pi
    x0_oscillator: tuple = (1.0, 0.0)

    # NEF/LIF
    tau_syn: float = 0.05
    tau_rc: float = 0.02
    tau_ref: float = 0.002
    rate_low: float = 100.0
    rate_high: float = 200.0

    # Training / learning
    cue_end: float = 5
    ridge_lambda_rec: float = 1e-2
    ridge_lambda_e: float = 1e-2
    eta: float = 1e-3


    # Fixed hidden populations
    n_lat: int = 50
    n_err: int = 50

    # Offline recurrent decoder for autonomous o1
    max_train_samples: int = 3000

    # Static decoders for PC-SC error population
    decoder_train_samples: int = 6000

    # Metrics
    sync_start_offset_after_cue: float = 0.5
    smooth_win: int = 101
    amp_thresh: float = 1e-3

    # Output / plotting
    out_dir: str = "results_PC-SC_PC-EC_10tests_8metrics_4x2_spikes"
    dpi: int = 260
    show_plots: bool = False
    save_raw_csv: bool = True
    save_raw_xlsx: bool = True
    save_mean_std_csv: bool = True
    save_mean_std_xlsx: bool = True
    save_big_4x2_figure: bool = True

    # Randomness
    base_seed: int = 1

    # Benchmark environment (reported for reproducibility)
    benchmark_os: str = "Windows 11 Pro 25H2"
    benchmark_cpu: str = "AMD Ryzen 7 7700, 8 cores / 16 logical processors, max reported clock 3.8 GHz"
    benchmark_ram: str = "32 GB DDR5-4800"
    benchmark_gpu: str = "NVIDIA GeForce RTX 4070 Ti SUPER (not used by the NumPy CPU simulations)"
    benchmark_storage: str = "XPG GAMMIX S70 BLADE 2 TB + WD_BLACK SN770 1 TB NVMe SSD"
    benchmark_ide: str = "Visual Studio Code 1.135.0"


CFG = Config()

mpl.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 20,
})


# =============================================================================
# Columns and metrics
# =============================================================================

# IMPORTANT:
# mean_phase_deg here is the mean absolute mean phase difference (in degrees)
# averaged over signal components. Using abs() avoids cancellation of positive
# and negative phase shifts across dimensions.
PLOT_METRICS = [
    ("sim_time_s", "Simulation time, s", False),
    ("spikes_per_s", "Emitted spikes per simulated second", False),

    ("mean_amp_corr", "Mean amplitude correlation", True),
    ("mean_freq_corr", "Mean frequency correlation", True),

    ("mean_plv", "Mean PLV", True),
    ("mean_phase_deg", "Mean phase difference, deg", False),

    ("rmse_ref_o1", r"RMSE(reference, $o_1$)", False),
    ("rmse_o1_g", r"RMSE($o_1$, $g$)", False),
]

SUMMARY_METRICS = [m[0] for m in PLOT_METRICS]

RAW_SAVE_COLUMNS = [
    "signal",
    "N_sens",
    "arch",
    "test_id",
    *SUMMARY_METRICS,
]

AGG_SAVE_COLUMNS = [
    "signal",
    "N_sens",
    "arch",
    "n_tests",
]
for _metric in SUMMARY_METRICS:
    AGG_SAVE_COLUMNS.extend([f"{_metric}_mean", f"{_metric}_std"])


# =============================================================================
# Timing
# =============================================================================

def measure_time(func, *args, **kwargs):
    """Wall-clock time of one call, with no profiler attached.

    Garbage collection is completed before the clock starts so that a pending
    collection from previous work is not charged to the measured simulation.
    """
    gc.collect()
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


# =============================================================================
# LIF / NEF helpers
# =============================================================================

def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return np.nan
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def current_from_rate(r, tau_rc=0.02, tau_ref=0.002):
    r = np.asarray(r, dtype=float)
    k = (1.0 / r - tau_ref) / tau_rc
    return 1.0 / (1.0 - np.exp(-k))


def lif_rate_from_current(J, tau_rc=0.02, tau_ref=0.002):
    J = np.asarray(J, dtype=float)
    out = np.zeros_like(J, dtype=float)
    mask = J > 1.0
    out[mask] = 1.0 / (tau_ref - tau_rc * np.log1p(-1.0 / J[mask]))
    return out


def init_population(N, D, rng, tau_rc, tau_ref, rate_low=100.0, rate_high=200.0):
    enc = rng.normal(size=(N, D))
    enc /= np.linalg.norm(enc, axis=1, keepdims=True) + 1e-12

    x_intercepts = rng.uniform(-1.0, 1.0, size=N)
    max_rates = rng.uniform(rate_low, rate_high, size=N)

    Jmax = current_from_rate(max_rates, tau_rc, tau_ref)
    gain = (Jmax - 1.0) / (1.0 - x_intercepts)
    bias = 1.0 - gain * x_intercepts
    return enc.astype(float), gain.astype(float), bias.astype(float)


def lif_population_step(V, ref_count, inp, enc, gain, bias, dt, tau_rc, tau_ref):
    J = gain * (enc @ inp) + bias
    active = ref_count <= 0

    V[active] += (dt / tau_rc) * (-V[active] + J[active])
    V[~active] = 0.0

    spikes = V >= 1.0
    V[spikes] = 0.0

    ref_count[ref_count > 0] -= 1
    ref_steps = max(1, int(round(tau_ref / dt)))
    ref_count[spikes] = ref_steps

    return V, ref_count, spikes


def update_filtered_activity(a, spikes, dt, tau_syn):
    return a + dt * (-a / tau_syn) + spikes.astype(float) / tau_syn


def population_rates(X, enc, gain, bias, tau_rc, tau_ref):
    J = X @ enc.T
    J = J * gain[None, :] + bias[None, :]
    return lif_rate_from_current(J, tau_rc=tau_rc, tau_ref=tau_ref)


def solve_ridge(A, Y, lam=1e-2):
    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    G = A.T @ A
    B = A.T @ Y
    # Регуляризация относительно масштаба активности, а не абсолютная.
    # Иначе фиксированная lam становится пренебрежимой при большом N ->
    # переобучение рекуррентного декодера -> срыв предельного цикла o1.
    reg = lam * np.mean(np.diag(G))
    return np.linalg.solve(G + reg * np.eye(G.shape[0]), B)


def solve_static_decoder(enc, gain, bias, D, rng, cfg: Config):
    X = rng.uniform(-1.0, 1.0, size=(cfg.decoder_train_samples, D))
    n_zero = min(cfg.decoder_train_samples, max(100, cfg.decoder_train_samples // 10))
    X[:n_zero] = 0.25 * rng.normal(size=(n_zero, D))
    X = np.clip(X, -1.0, 1.0)
    A = population_rates(X, enc, gain, bias, cfg.tau_rc, cfg.tau_ref)
    return solve_ridge(A, X, cfg.ridge_lambda_e)


# =============================================================================
# Reference dynamics
# =============================================================================

def lorenz_rk4_trajectory(T, dt, x0, sigma=10.0, rho=28.0, beta=2.667):
    t = np.arange(0.0, T + dt, dt)
    raw = np.zeros((t.size, 3), dtype=float)
    raw[0] = np.asarray(x0, dtype=float)

    def f(s):
        x, y, z = s
        return np.array([
            sigma * (y - x),
            x * rho - x * z - y,
            x * y - beta * z,
        ], dtype=float)

    s = raw[0].copy()
    for n in range(1, t.size):
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        raw[n] = s

    scale = np.max(np.abs(raw), axis=0)
    ref = raw / (scale + 1e-12)
    return t, raw, ref


def oscillator_trajectory(T, dt, omega=2.0 * math.pi, x0=(1.0, 0.0)):
    t = np.arange(0.0, T + dt, dt)
    theta = omega * dt
    M = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ], dtype=float)

    u = np.zeros((t.size, 2), dtype=float)
    u[0] = np.asarray(x0, dtype=float)
    u[0] /= np.linalg.norm(u[0]) + 1e-12

    for n in range(1, t.size):
        u[n] = M @ u[n - 1]

    return t, u.copy(), u.copy()


def make_reference_signal(signal_name, cfg: Config):
    name = signal_name.lower().strip()

    if name == "lorenz":
        t, raw, ref = lorenz_rk4_trajectory(
            cfg.T,
            cfg.dt,
            cfg.x0_lorenz,
            sigma=cfg.sigma,
            rho=cfg.rho,
            beta=cfg.beta,
        )
        return {
            "signal": "lorenz",
            "title": "Lorenz attractor",
            "t": t,
            "raw": raw,
            "ref": ref,
            "D": 3,
            "labels": ["x", "y", "z"],
        }

    if name == "oscillator":
        t, raw, ref = oscillator_trajectory(
            cfg.T,
            cfg.dt,
            omega=cfg.oscillator_omega,
            x0=cfg.x0_oscillator,
        )
        return {
            "signal": "oscillator",
            "title": "Ordinary oscillator",
            "t": t,
            "raw": raw,
            "ref": ref,
            "D": 2,
            "labels": ["x", "y"],
        }

    raise ValueError(f"Unknown signal_name={signal_name!r}. Use 'lorenz' or 'oscillator'.")


# =============================================================================
# Synchronization metrics
# =============================================================================

def analytic_signal(x):
    x = np.asarray(x, dtype=float)
    N = x.size
    Xf = np.fft.fft(x)
    H = np.zeros(N)
    if N % 2 == 0:
        H[0] = 1.0
        H[N // 2] = 1.0
        H[1:N // 2] = 2.0
    else:
        H[0] = 1.0
        H[1:(N + 1) // 2] = 2.0
    return np.fft.ifft(Xf * H)


def moving_average(x, win):
    x = np.asarray(x, dtype=float)
    k = int(win)
    if k <= 1 or x.size <= 1:
        return x
    # np.convolve(..., mode="same") returns max(len(x), len(kernel)); limiting
    # the window prevents a length mismatch for very short test trajectories.
    k = min(k, x.size)
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")


def sync_metrics_1d(ref, rec, dt, smooth_win=1001, amp_thresh=1e-3):
    ref = np.asarray(ref, dtype=float)
    rec = np.asarray(rec, dtype=float)

    ref_c = ref - np.mean(ref)
    rec_c = rec - np.mean(rec)

    z_ref = analytic_signal(ref_c)
    z_rec = analytic_signal(rec_c)

    A_ref = np.abs(z_ref)
    A_rec = np.abs(z_rec)
    amp_corr = safe_corr(A_ref, A_rec)

    mask = (A_ref > amp_thresh) & (A_rec > amp_thresh)

    phi_ref = np.unwrap(np.angle(z_ref))
    phi_rec = np.unwrap(np.angle(z_rec))

    f_ref = np.diff(phi_ref) / (2.0 * np.pi * dt)
    f_rec = np.diff(phi_rec) / (2.0 * np.pi * dt)

    if smooth_win is not None and smooth_win > 1:
        f_ref = moving_average(f_ref, smooth_win)
        f_rec = moving_average(f_rec, smooth_win)

    mask_f = mask[1:]
    if np.sum(mask_f) > 2:
        freq_corr = safe_corr(f_ref[mask_f], f_rec[mask_f])
    else:
        freq_corr = np.nan

    dphi = (phi_ref - phi_rec)[mask]
    if dphi.size:
        c = np.mean(np.exp(1j * dphi))
        plv = float(np.abs(c))
        mean_phase_diff = float(np.angle(c))
    else:
        plv = np.nan
        mean_phase_diff = np.nan

    return amp_corr, freq_corr, plv, mean_phase_diff


def summarize_sync(u, g, ref, t, cfg: Config):
    start = cfg.cue_end + cfg.sync_start_offset_after_cue
    idx = int(np.searchsorted(t, start))
    if idx >= len(t) - 4:
        idx = max(0, len(t) // 2)

    amp_vals = []
    freq_vals = []
    plv_vals = []
    phase_deg_vals = []

    for d in range(ref.shape[1]):
        a, f, p, ph = sync_metrics_1d(
            u[idx:, d],
            g[idx:, d],
            cfg.dt,
            smooth_win=cfg.smooth_win,
            amp_thresh=cfg.amp_thresh,
        )
        amp_vals.append(a)
        freq_vals.append(f)
        plv_vals.append(p)

        if np.isfinite(ph):
            phase_deg_vals.append(np.abs(np.degrees(ph)))
        else:
            phase_deg_vals.append(np.nan)

    return {
        "mean_amp_corr": float(np.nanmean(amp_vals)),
        "mean_freq_corr": float(np.nanmean(freq_vals)),
        "mean_plv": float(np.nanmean(plv_vals)),
        "mean_phase_deg": float(np.nanmean(phase_deg_vals)),
        "rmse_ref_o1": float(np.sqrt(np.mean((ref[idx:] - u[idx:]) ** 2))),
        "rmse_o1_g": float(np.sqrt(np.mean((u[idx:] - g[idx:]) ** 2))),
    }


# =============================================================================
# Training recurrent sensory decoder and running architectures
# =============================================================================

def make_train_indices(N_t, max_train_samples):
    if N_t <= max_train_samples:
        return np.arange(N_t, dtype=int)
    idx = np.linspace(0, N_t - 1, max_train_samples, dtype=int)
    return np.unique(idx)


def train_sensory_recurrent_decoder(ref, rec_target, enc_s, gain_s, bias_s, cfg: Config):
    N_t = ref.shape[0]
    N_sens = enc_s.shape[0]
    train_idx = make_train_indices(N_t, cfg.max_train_samples)

    A_train = np.zeros((train_idx.size, N_sens), dtype=np.float32)
    Y_train = rec_target[train_idx].astype(np.float32)

    V = np.zeros(N_sens, dtype=float)
    ref_count = np.zeros(N_sens, dtype=int)
    a = np.zeros(N_sens, dtype=float)

    j = 0
    for n in range(N_t):
        V, ref_count, spikes = lif_population_step(
            V,
            ref_count,
            ref[n],
            enc_s,
            gain_s,
            bias_s,
            cfg.dt,
            cfg.tau_rc,
            cfg.tau_ref,
        )
        a = update_filtered_activity(a, spikes, cfg.dt, cfg.tau_syn)
        if j < train_idx.size and n == train_idx[j]:
            A_train[j] = a.astype(np.float32)
            j += 1

    return solve_ridge(A_train, Y_train, cfg.ridge_lambda_rec)


def prepare_architecture(arch, D, cfg: Config, seed):
    """Create fixed populations/decoders before benchmark timing starts.

    In particular, the PC-SC static error decoder is fitted here, outside the
    measured online simulation. This keeps sim_time_s free of offline decoder
    fitting and population initialization costs.
    """
    arch = arch.upper()
    if arch not in {"PC-EC", "PC-SC"}:
        raise ValueError("arch must be 'PC-EC' or 'PC-SC'")

    rng = np.random.default_rng(seed)

    enc_z, gain_z, bias_z = init_population(
        cfg.n_lat,
        D,
        rng,
        cfg.tau_rc,
        cfg.tau_ref,
        cfg.rate_low,
        cfg.rate_high,
    )

    if arch == "PC-SC":
        enc_e, gain_e, bias_e = init_population(
            cfg.n_err,
            D,
            rng,
            cfg.tau_rc,
            cfg.tau_ref,
            cfg.rate_low,
            cfg.rate_high,
        )
        W_e = solve_static_decoder(enc_e, gain_e, bias_e, D, rng, cfg)
    else:
        enc_e = gain_e = bias_e = W_e = None

    return {
        "enc_z": enc_z,
        "gain_z": gain_z,
        "bias_z": bias_z,
        "enc_e": enc_e,
        "gain_e": gain_e,
        "bias_e": bias_e,
        "W_e": W_e,
    }


def run_architecture(
    arch,
    signal_info,
    W_s_rec,
    enc_s,
    gain_s,
    bias_s,
    arch_params,
    cfg: Config,
):
    """Run only the online SNN dynamics and PES adaptation.

    Offline recurrent/static decoder fitting and population initialization are
    deliberately excluded from this function so that the measured sim_time_s
    corresponds to the online simulation itself.
    """
    arch = arch.upper()
    if arch not in {"PC-EC", "PC-SC"}:
        raise ValueError("arch must be 'PC-EC' or 'PC-SC'")

    ref = signal_info["ref"]
    t = signal_info["t"]
    D = ref.shape[1]
    N_t = t.size
    N_sens = enc_s.shape[0]
    use_error_population = arch == "PC-SC"

    enc_z = arch_params["enc_z"]
    gain_z = arch_params["gain_z"]
    bias_z = arch_params["bias_z"]
    enc_e = arch_params["enc_e"]
    gain_e = arch_params["gain_e"]
    bias_e = arch_params["bias_e"]
    W_e = arch_params["W_e"]

    u = np.zeros((N_t, D), dtype=float)
    z = np.zeros((N_t, D), dtype=float)
    g = np.zeros((N_t, D), dtype=float)

    u[0] = ref[0]
    g_hat = np.zeros(D, dtype=float)

    V_s = np.zeros(N_sens, dtype=float)
    ref_s = np.zeros(N_sens, dtype=int)
    a_s = np.zeros(N_sens, dtype=float)

    V_z = np.zeros(cfg.n_lat, dtype=float)
    ref_z = np.zeros(cfg.n_lat, dtype=int)
    a_z = np.zeros(cfg.n_lat, dtype=float)

    if use_error_population:
        V_e = np.zeros(cfg.n_err, dtype=float)
        ref_e = np.zeros(cfg.n_err, dtype=int)
        a_e = np.zeros(cfg.n_err, dtype=float)
    else:
        V_e = ref_e = a_e = None

    W_pred = np.zeros((cfg.n_lat, D), dtype=float)

    # Spike counters for the ONLINE simulation only.
    # PC-EC: sensory + latent populations.
    # PC-SC: sensory + error + latent populations.
    n_spikes_sens = 0
    n_spikes_err = 0
    n_spikes_lat = 0

    for n in range(1, N_t):
        # 1) Autonomous sensory population o1
        V_s, ref_s, spikes_s = lif_population_step(
            V_s,
            ref_s,
            u[n - 1],
            enc_s,
            gain_s,
            bias_s,
            cfg.dt,
            cfg.tau_rc,
            cfg.tau_ref,
        )
        n_spikes_sens += int(np.count_nonzero(spikes_s))

        a_s = update_filtered_activity(a_s, spikes_s, cfg.dt, cfg.tau_syn)
        rec_s = a_s @ W_s_rec

        if t[n] < cfg.cue_end:
            cue = ref[n - 1] - u[n - 1]
        else:
            cue = np.zeros(D)

        u_dot = (-u[n - 1] + rec_s + cue) / cfg.tau_syn
        u[n] = u[n - 1] + cfg.dt * u_dot

        # 2) Prediction error
        e = u[n] - g_hat

        # 3) Error pathway into latent state z
        if use_error_population:
            V_e, ref_e, spikes_e = lif_population_step(
                V_e,
                ref_e,
                np.clip(e, -1.0, 1.0),
                enc_e,
                gain_e,
                bias_e,
                cfg.dt,
                cfg.tau_rc,
                cfg.tau_ref,
            )
            n_spikes_err += int(np.count_nonzero(spikes_e))

            a_e = update_filtered_activity(a_e, spikes_e, cfg.dt, cfg.tau_syn)
            e_drive = a_e @ W_e
        else:
            e_drive = e

        z_dot = (-z[n - 1] / cfg.tau_syn) + e_drive
        z[n] = z[n - 1] + cfg.dt * z_dot
        z[n] = np.clip(z[n], -1.0, 1.0)

        # 4) Latent population o2/z and top-down decoder
        V_z, ref_z, spikes_z = lif_population_step(
            V_z,
            ref_z,
            z[n],
            enc_z,
            gain_z,
            bias_z,
            cfg.dt,
            cfg.tau_rc,
            cfg.tau_ref,
        )
        n_spikes_lat += int(np.count_nonzero(spikes_z))

        a_z = update_filtered_activity(a_z, spikes_z, cfg.dt, cfg.tau_syn)
        g_raw = a_z @ W_pred
        g_hat = g_hat + (cfg.dt / cfg.tau_syn) * (g_raw - g_hat)
        g[n] = g_hat

        # 5) PES-like local decoder update
        W_pred += cfg.eta * np.outer(a_z, e) * cfg.dt

    simulated_duration_s = (N_t - 1) * cfg.dt
    total_spikes = n_spikes_sens + n_spikes_err + n_spikes_lat
    spikes_per_s = total_spikes / simulated_duration_s if simulated_duration_s > 0 else np.nan

    return {
        "signal": signal_info["signal"],
        "arch": arch,
        "N_sens": N_sens,
        "D": D,
        "u": u,
        "g": g,
        "total_spikes": int(total_spikes),
        "spikes_per_s": float(spikes_per_s),
        "spike_counts": {
            "sensory": int(n_spikes_sens),
            "error": int(n_spikes_err),
            "latent": int(n_spikes_lat),
        },
    }


# =============================================================================
# Saving selected columns only
# =============================================================================

def keep_existing_columns(df, columns):
    existing = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"Warning: missing columns skipped: {missing}")
    return df[existing].copy()


def build_mean_std_table(df_raw):
    group_cols = ["signal", "N_sens", "arch"]
    grouped = df_raw.groupby(group_cols, as_index=False)

    mean_df = grouped[SUMMARY_METRICS].mean()
    std_df = grouped[SUMMARY_METRICS].std(ddof=1)
    count_df = grouped.size().rename(columns={"size": "n_tests"})

    out = count_df.copy()
    for metric in SUMMARY_METRICS:
        out[f"{metric}_mean"] = mean_df[metric]
        out[f"{metric}_std"] = std_df[metric]

    return keep_existing_columns(out, AGG_SAVE_COLUMNS)


def save_results(df_raw_full, cfg: Config, out_dir: Path):
    df_raw = keep_existing_columns(df_raw_full, RAW_SAVE_COLUMNS)
    df_agg = build_mean_std_table(df_raw)

    if cfg.save_raw_csv:
        df_raw.to_csv(out_dir / "summary_metrics_raw.csv", index=False, sep=";", decimal=",")
    if cfg.save_raw_xlsx:
        df_raw.to_excel(out_dir / "summary_metrics_raw.xlsx", index=False)

    if cfg.save_mean_std_csv:
        df_agg.to_csv(out_dir / "summary_metrics_mean_std.csv", index=False, sep=";", decimal=",")
    if cfg.save_mean_std_xlsx:
        df_agg.to_excel(out_dir / "summary_metrics_mean_std.xlsx", index=False)

    return df_raw, df_agg


# =============================================================================
# One big 4x2 mean ± std figure
# =============================================================================

def add_panel_label(ax, label):
    ax.text(
        0.02,
        0.96,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
    )


def mean_std_by_arch_and_n(df_sig, arch, metric):
    g_arch = df_sig[df_sig["arch"] == arch].copy()
    xs, means, stds = [], [], []

    for N_sens, g in g_arch.groupby("N_sens"):
        xs.append(N_sens)
        means.append(g[metric].mean())
        stds.append(g[metric].std(ddof=1))

    order = np.argsort(xs)
    xs = np.asarray(xs, dtype=float)[order]
    means = np.asarray(means, dtype=float)[order]
    stds = np.asarray(stds, dtype=float)[order]
    return xs, means, stds


def plot_big_4x2_mean_std(df_raw, cfg: Config, fig_dir: Path):
    panel_labels = ["a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)"]

    for signal_name in sorted(df_raw["signal"].unique()):
        df_sig = df_raw[df_raw["signal"] == signal_name].copy()
        if df_sig.empty:
            continue

        fig, axes = plt.subplots(4, 2, figsize=(18, 22))
        axes = axes.ravel()

        # Metric order already arranged so RMSE metrics are in the last row
        for ax, panel_label, (metric, ylabel, ylim01) in zip(axes, panel_labels, PLOT_METRICS):
            for arch in sorted(df_sig["arch"].unique()):
                xs, means, stds = mean_std_by_arch_and_n(df_sig, arch, metric)
                if xs.size == 0:
                    continue

                ax.plot(xs, means, marker="o", linewidth=2.2, label=arch)
                ax.fill_between(xs, means - stds, means + stds, alpha=0.18)

            add_panel_label(ax, panel_label)
            ax.set_title(ylabel)
            ax.set_xlabel(r"$N_{sens}$")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.35)

            if ylim01:
                ax.set_ylim(-0.05, 1.05)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=max(1, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
            fontsize=14,
        )

        fig.suptitle(
            f"{signal_name}: PC-SC vs PC-EC, mean ± std over {cfg.n_tests} tests",
            y=0.998,
            fontsize=22,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.985])

        out_png = fig_dir / f"mean_std_{signal_name}_8metrics_4x2.png"
        out_pdf = fig_dir / f"mean_std_{signal_name}_8metrics_4x2.pdf"
        fig.savefig(out_png, dpi=cfg.dpi, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")

        if cfg.show_plots:
            plt.show()
        plt.close(fig)

        print(f"Saved 4x2 figure: {out_png}")
        print(f"Saved 4x2 figure: {out_pdf}")


# =============================================================================
# Benchmark environment
# =============================================================================

def save_benchmark_environment(cfg: Config, out_dir: Path):
    """Save the exact benchmark environment reported in the manuscript."""
    lines = [
        f"Operating system: {cfg.benchmark_os}",
        f"CPU: {cfg.benchmark_cpu}",
        f"RAM: {cfg.benchmark_ram}",
        f"GPU: {cfg.benchmark_gpu}",
        f"Storage: {cfg.benchmark_storage}",
        f"IDE: {cfg.benchmark_ide}",
        f"Python: {platform.python_version()}",
        f"NumPy: {np.__version__}",
        f"pandas: {pd.__version__}",
        f"Matplotlib: {mpl.__version__}",
        "Benchmark device: CPU",
        "GPU acceleration: no",
        "sim_time_s includes: online SNN simulation + PES adaptation",
        "sim_time_s excludes: sensory recurrent decoder fitting, PC-SC static error-decoder fitting, population initialization, and metric post-processing",
        "spikes_per_s: total online spikes from all spiking populations divided by simulated duration",
    ]
    path = out_dir / "benchmark_environment.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# =============================================================================
# Main experiment
# =============================================================================

def run_sweep(cfg: Config):
    out_dir = Path(cfg.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    environment_path = save_benchmark_environment(cfg, out_dir)

    with open(out_dir / "config.txt", "w", encoding="utf-8") as f:
        for k, v in asdict(cfg).items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nRAW_SAVE_COLUMNS: {RAW_SAVE_COLUMNS}\n")
        f.write(f"AGG_SAVE_COLUMNS: {AGG_SAVE_COLUMNS}\n")
        f.write("\nPLOT_METRICS:\n")
        for metric, label, _ in PLOT_METRICS:
            f.write(f"  {metric}: {label}\n")

    rows = []

    print("\n=== PC-SC vs PC-EC sweep: runtime + spikes/s + synchronization metrics ===")
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Signals: {cfg.signal_names}")
    print(f"N_sens values: {cfg.n_sens_values}")
    print(f"Tests per condition: {cfg.n_tests}")
    print(f"Metrics: {SUMMARY_METRICS}\n")

    for signal_name in cfg.signal_names:
        signal_info = make_reference_signal(signal_name, cfg)
        ref = signal_info["ref"]
        D = signal_info["D"]

        dref_dt = np.gradient(ref, cfg.dt, axis=0)
        rec_target = ref + cfg.tau_syn * dref_dt

        print(f"\n################ {signal_info['title']} | D={D} ################")

        for N_sens in cfg.n_sens_values:
            print(f"\n================ {signal_name}, N_sens={N_sens} ================")

            for test_id in range(cfg.n_tests):
                print(f"\n--- test {test_id + 1}/{cfg.n_tests} ---")

                # New sensory population and recurrent decoder for every test.
                # Decoder fitting is OFFLINE and is excluded from sim_time_s.
                sens_seed = cfg.base_seed + 1_000_000 * test_id + 100_000 + N_sens + 999 * D
                rng_s = np.random.default_rng(sens_seed)

                enc_s, gain_s, bias_s = init_population(
                    N_sens,
                    D,
                    rng_s,
                    cfg.tau_rc,
                    cfg.tau_ref,
                    cfg.rate_low,
                    cfg.rate_high,
                )

                W_s_rec, sensory_decoder_fit_time_s = measure_time(
                    train_sensory_recurrent_decoder,
                    ref,
                    rec_target,
                    enc_s,
                    gain_s,
                    bias_s,
                    cfg,
                )

                print(
                    f"Offline sensory decoder fit: {sensory_decoder_fit_time_s:.2f}s "
                    f"(excluded from sim_time_s)"
                )

                for arch_idx, arch in enumerate(["PC-EC", "PC-SC"]):
                    sim_seed = (
                        cfg.base_seed
                        + 1_000_000 * test_id
                        + 10_000 * arch_idx
                        + N_sens
                        + 999 * D
                    )

                    # All fixed architecture initialization, including the PC-SC
                    # static error decoder, is completed BEFORE the timer starts.
                    arch_params = prepare_architecture(arch, D, cfg, sim_seed)

                    result, sim_time_s = measure_time(
                        run_architecture,
                        arch,
                        signal_info,
                        W_s_rec,
                        enc_s,
                        gain_s,
                        bias_s,
                        arch_params,
                        cfg,
                    )

                    # Synchronization/RMSE post-processing is deliberately
                    # performed AFTER the timer and is not part of sim_time_s.
                    m = summarize_sync(
                        result["u"],
                        result["g"],
                        signal_info["ref"],
                        signal_info["t"],
                        cfg,
                    )

                    row = {
                        "signal": signal_info["signal"],
                        "N_sens": N_sens,
                        "arch": arch,
                        "test_id": test_id + 1,
                        "sim_time_s": sim_time_s,
                        "spikes_per_s": result["spikes_per_s"],
                        "mean_amp_corr": m["mean_amp_corr"],
                        "mean_freq_corr": m["mean_freq_corr"],
                        "mean_plv": m["mean_plv"],
                        "mean_phase_deg": m["mean_phase_deg"],
                        "rmse_ref_o1": m["rmse_ref_o1"],
                        "rmse_o1_g": m["rmse_o1_g"],
                    }
                    rows.append(row)

                    print(
                        f"{arch}: "
                        f"time={row['sim_time_s']:.2f}s, "
                        f"spikes/s={row['spikes_per_s']:.1f}, "
                        f"amp={row['mean_amp_corr']:.3f}, "
                        f"freq={row['mean_freq_corr']:.3f}, "
                        f"PLV={row['mean_plv']:.3f}, "
                        f"phase_deg={row['mean_phase_deg']:.3f}, "
                        f"RMSE(ref,o1)={row['rmse_ref_o1']:.6f}, "
                        f"RMSE(o1,g)={row['rmse_o1_g']:.6f}"
                    )

                    del result, arch_params
                    gc.collect()

                # Save partial results after each test
                df_partial_full = pd.DataFrame(rows)
                save_results(df_partial_full, cfg, out_dir)

                del W_s_rec, enc_s, gain_s, bias_s
                gc.collect()

    df_full = pd.DataFrame(rows)
    df_raw, df_agg = save_results(df_full, cfg, out_dir)

    print("\n=== Raw selected table ===")
    print(df_raw.to_string(index=False))

    print("\n=== Mean ± std selected table ===")
    print(df_agg.to_string(index=False))

    if cfg.save_big_4x2_figure:
        plot_big_4x2_mean_std(df_raw, cfg, fig_dir)

    print(f"\nBenchmark environment: {environment_path}")
    if cfg.save_raw_csv:
        print(f"Saved raw CSV:       {out_dir / 'summary_metrics_raw.csv'}")
    if cfg.save_raw_xlsx:
        print(f"Saved raw XLSX:      {out_dir / 'summary_metrics_raw.xlsx'}")
    if cfg.save_mean_std_csv:
        print(f"Saved mean±std CSV:  {out_dir / 'summary_metrics_mean_std.csv'}")
    if cfg.save_mean_std_xlsx:
        print(f"Saved mean±std XLSX: {out_dir / 'summary_metrics_mean_std.xlsx'}")
    if cfg.save_big_4x2_figure:
        print(f"Saved figures:       {fig_dir}")

    return df_raw, df_agg


if __name__ == "__main__":


    run_sweep(CFG)


