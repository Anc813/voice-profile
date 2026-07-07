#!/usr/bin/env python3
"""
v10 — v9's analysis, with an optional v6-style scrolling renderer.

Two render modes share ONE analysis path, so they can never drift the way
v6 drifted from v9 (v6 had a weaker HNR, no CPP, linear-frequency tilt,
all-frame formants and a 75 Hz floor):

  default            static dashboard PNG per file -> 1 fps MP4. Fast.
                     Byte-for-byte the same output as v9.

  --scroll           v6-style animation: a time-linear strip scrolls under
                     a centred playhead at 30 fps (short files fit one frame
                     and the playhead sweeps L->R instead). Same v9 metrics.
                     MUCH slower to render (full-res strip + per-frame ffmpeg
                     overlay) — use it only when you actually want motion.

Analysis (unchanged from v9): HNR (Get mean, floor-matched), CPP, %voiced,
median + P5-P95 pitch, voiced-only formants, dB/oct spectral tilt, 50 Hz floor.
"""

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import parselmouth
from parselmouth.praat import call

HERE = Path(__file__).parent.resolve()
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".opus", ".ogg", ".flac", ".aac"}

# pitch analysis range (Hz) — 50 Hz floor captures low male voices / vocal fry
PITCH_FLOOR = 50.0
PITCH_CEILING = 500.0

# ── dashboard styling (static mode) ──────────────────────────────────────────
plt.rcParams["font.monospace"] = [
    "Consolas", "Cascadia Mono", "Cascadia Code", "DejaVu Sans Mono",
]

COL_LABEL = "#5a5a5a"   # metric name
COL_VALUE = "#1a1a1a"   # neutral value
COL_NA = "#8a8a8a"      # N/A
STATS_FONTSIZE = 9.0

# shared plot colours
C_WAVE = "#95a5a6"
C_INT = "#2ecc71"
C_PITCH = "#00d4ff"
C_F1, C_F2, C_F3 = "#e6194B", "#FF00FF", "#ffd700"


def _pitch_register_color(f0: float | None) -> str:
    """Informational colour by vocal register (not a good/bad judgement)."""
    if not f0:
        return COL_NA
    if f0 < 150:
        return "#2f86c5"   # male — blue
    if f0 < 200:
        return "#8e44ad"   # mid / androgynous — purple
    return "#d84a6b"       # female / child — rose


def _cpp_traffic_color(v: float | None) -> str:
    """Traffic-light for CPP only — robust enough to reverb to be trustworthy."""
    if v is None:
        return COL_NA
    if v > 10.0:
        return "#1e8449"   # green — strong harmonic support
    if v >= 6.0:
        return "#c8860a"   # amber — moderate noise / reverb
    return "#c0392b"       # red — heavily noisy / smeared


# ── voice metrics (identical to v9) ─────────────────────────────────────────

def _hnr(sound: parselmouth.Sound) -> float | None:
    """
    Mean HNR (dB) over defined (periodic) frames via Praat's own "Get mean":
    it averages only frames that are not undefined (excludes silence) while
    keeping genuinely negative dB values, and avoids the grid-mismatch trap of
    resampling harmonicity at the pitch track's frame times (Praat stores
    undefined frames as a large negative sentinel, not NaN).

    minimum_pitch tied to PITCH_FLOOR: the 75 Hz default drops low male voices.
    """
    try:
        h = sound.to_harmonicity(time_step=0.01, minimum_pitch=PITCH_FLOOR)
        mean = call(h, "Get mean", 0, 0)  # over defined frames only
        if mean is None or not np.isfinite(mean):
            return None
        return float(mean)
    except Exception:
        return None


def _cpps(sound: parselmouth.Sound) -> float | None:
    """Smoothed Cepstral Peak Prominence (dB) — reverb-tolerant periodicity."""
    try:
        pc = call(sound, "To PowerCepstrogram", 60.0, 0.002, 5000.0, 50.0)
        return float(call(
            pc, "Get CPPS",
            True,          # subtract tilt before smoothing
            0.02,          # time averaging window (s)
            0.0005,        # quefrency averaging window (s)
            PITCH_FLOOR,   # peak search from (Hz)
            PITCH_CEILING, # peak search to (Hz)
            0.05,          # tolerance
            "parabolic",   # interpolation
            0.001,         # tilt line quefrency start (s)
            0.0,           # tilt line quefrency end (0 = to end)
            "Exponential decay",
            "Robust",      # fit method
        ))
    except Exception:
        return None


def _spectral_tilt_voiced(sound: parselmouth.Sound, pitch: parselmouth.Pitch) -> float | None:
    """
    Spectral tilt in dB/octave from an LTAS built over voiced frames.

    A 40 ms Hann-windowed FFT around every voiced pitch frame; power spectra
    averaged, then a line fit on the log10-frequency axis over 100-5000 Hz.
    Slope per decade converted to dB/octave.
    """
    vals = sound.values[0]  # mono
    sr = int(sound.sampling_frequency)
    t0 = float(sound.xs()[0])

    freqs_p = pitch.selected_array["frequency"]
    times_p = pitch.xs()

    win_len = int(round(0.04 * sr))  # 40 ms
    if win_len < 32:
        return None
    if win_len % 2:
        win_len += 1
    half = win_len // 2
    window = np.hanning(win_len)

    acc = None
    count = 0
    for t, f in zip(times_p, freqs_p):
        if f <= 0:
            continue
        center = int(round((t - t0) * sr))
        s, e = center - half, center + half
        if s < 0 or e > len(vals):
            continue
        seg = vals[s:e] * window
        power = np.abs(np.fft.rfft(seg)) ** 2
        acc = power if acc is None else acc + power
        count += 1

    if count == 0:
        return None

    avg = acc / count
    fft_freqs = np.fft.rfftfreq(win_len, d=1.0 / sr)
    mask = (fft_freqs >= 100) & (fft_freqs <= 5000)
    if mask.sum() < 2:
        return None

    db = 10 * np.log10(avg[mask] + 1e-10)
    slope_per_decade, _ = np.polyfit(np.log10(fft_freqs[mask]), db, 1)
    return float(slope_per_decade * np.log10(2.0))  # dB/octave


def extract_voice_metrics(sound: parselmouth.Sound, pitch: parselmouth.Pitch) -> dict:
    """Return jitter, shimmer, HNR, CPP, tilt_slope or None for each."""
    m: dict[str, float | None] = {
        "jitter": None, "shimmer": None, "hnr": None, "cpp": None, "tilt_slope": None,
    }

    try:
        point_process = call(sound, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
        jitter = call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        m["jitter"] = (jitter * 100) if jitter else 0.0
        shimmer = call([sound, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
        m["shimmer"] = (shimmer * 100) if shimmer else 0.0
    except Exception:
        pass

    m["hnr"] = _hnr(sound)
    m["cpp"] = _cpps(sound)
    try:
        m["tilt_slope"] = _spectral_tilt_voiced(sound, pitch)
    except Exception:
        pass

    return m


# ── audio helpers ───────────────────────────────────────────────────────────

def ensure_wav(audio_path: Path) -> tuple[Path, bool]:
    """Decode to WAV if parselmouth can't read the format directly."""
    if audio_path.suffix.lower() in (".wav", ".mp3"):
        return audio_path, False
    wav = audio_path.with_suffix(".wav")
    if not wav.exists():
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(audio_path),
            "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", str(wav),
        ], check=True, timeout=300)
    return wav, True


# ── shared analysis ─────────────────────────────────────────────────────────

def analyse(sound: parselmouth.Sound) -> dict:
    """One analysis pass feeding both renderers (v9 metrics, voiced-only formants)."""
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    spectrogram = sound.to_spectrogram(window_length=0.03, time_step=0.003)
    intensity = sound.to_intensity(time_step=0.01)
    metrics = extract_voice_metrics(sound, pitch)

    raw_pitch = pitch.selected_array["frequency"]
    voiced_ratio = float(np.mean(raw_pitch > 0)) if len(raw_pitch) else 0.0
    pitch_y = raw_pitch.copy()
    pitch_y[pitch_y == 0] = np.nan

    sg_db = 10 * np.log10(spectrogram.values + 1e-10)
    vmax = float(sg_db.max())
    vmin = vmax - 50

    # formants — voiced frames only (skip pauses / unvoiced consonants)
    f_t, f1, f2, f3 = [], [], [], []
    try:
        formant = sound.to_formant_burg(time_step=0.01)
        for t in formant.xs():
            fv = pitch.get_value_at_time(t)
            if not (fv and fv > 0):
                continue
            v1 = formant.get_value_at_time(1, t, parselmouth.FormantUnit.HERTZ)
            v2 = formant.get_value_at_time(2, t, parselmouth.FormantUnit.HERTZ)
            v3 = formant.get_value_at_time(3, t, parselmouth.FormantUnit.HERTZ)
            if v1 and v2 and v3:
                f_t.append(t); f1.append(v1); f2.append(v2); f3.append(v3)
    except Exception:
        pass

    return {
        "duration": sound.duration,
        "wave_x": sound.xs(), "wave_y": sound.values[0], "wave_yT": sound.values.T,
        "int_x": intensity.xs(), "int_y": intensity.values.T.ravel(), "int_yT": intensity.values.T,
        "sg_X": spectrogram.x_grid(), "sg_Y": spectrogram.y_grid(),
        "sg_db": sg_db, "vmin": vmin, "vmax": vmax,
        "pitch_x": pitch.xs(), "pitch_y": pitch_y,
        "f_t": np.array(f_t), "f1": np.array(f1), "f2": np.array(f2), "f3": np.array(f3),
        "metrics": metrics, "voiced_ratio": voiced_ratio,
    }


# ══════════════════════════════════════════════════════════════════════════
#  STATIC MODE  (v9 dashboard)
# ══════════════════════════════════════════════════════════════════════════

def _draw_stats_grid(ax, metrics, voiced_ratio, valid_pitches):
    """Render the metric dashboard as a 3-column grid on ``ax`` (axes-frac)."""
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.005, 0.06), 0.99, 0.88,
        boxstyle="round,pad=0.004,rounding_size=0.02",
        transform=ax.transAxes,
        facecolor="#eef0f3", edgecolor="#d3d7dd", linewidth=0.8,
        alpha=0.75, zorder=0, mutation_aspect=0.25,
    ))

    if len(valid_pitches) > 0:
        mean_p = np.mean(valid_pitches)
        sd_p = np.std(valid_pitches)
        med_p = np.median(valid_pitches)
        p5, p95 = np.percentile(valid_pitches, [5, 95])
    else:
        mean_p = sd_p = med_p = p5 = p95 = 0

    hnr_s = f"{metrics['hnr']:.0f} dB" if metrics["hnr"] is not None else "N/A"
    cpp_s = f"{metrics['cpp']:.1f} dB" if metrics["cpp"] is not None else "N/A"
    jit_s = f"{metrics['jitter']:.2f}%" if metrics["jitter"] is not None else "N/A"
    shim_s = f"{metrics['shimmer']:.2f}%" if metrics["shimmer"] is not None else "N/A"
    tilt_s = f"{metrics['tilt_slope']:.1f} dB/oct" if metrics["tilt_slope"] is not None else "N/A"

    cells = [
        (0, 0, "Pitch:",   f"{mean_p:.0f} Hz",            _pitch_register_color(mean_p)),
        (0, 1, "Median:",  f"{med_p:.0f} ±{sd_p:.0f}", COL_VALUE),
        (0, 2, "Range:",   f"{p5:.0f}-{p95:.0f} Hz",      COL_VALUE),
        (1, 0, "HNR:",     hnr_s,                          COL_VALUE),
        (1, 1, "CPP:",     cpp_s,                          _cpp_traffic_color(metrics["cpp"])),
        (1, 2, "Tilt:",    tilt_s,                         COL_VALUE),
        (2, 0, "Jitter:",  jit_s,                          COL_VALUE),
        (2, 1, "Shimmer:", shim_s,                         COL_VALUE),
        (2, 2, "Voiced:",  f"{voiced_ratio * 100:.0f}%",  COL_VALUE),
    ]

    col_x = [0.02, 0.225, 0.43]
    row_y = [0.78, 0.50, 0.22]
    value_dx = 0.072

    for col, row, label, value, vcolor in cells:
        x, y = col_x[col], row_y[row]
        ax.text(x, y, label, transform=ax.transAxes, ha="left", va="center",
                color=COL_LABEL, fontsize=STATS_FONTSIZE, family="monospace")
        ax.text(x + value_dx, y, value, transform=ax.transAxes, ha="left", va="center",
                color=vcolor, fontsize=STATS_FONTSIZE, family="monospace", fontweight="bold")


def render_static(audio_path: Path, data: dict) -> Path:
    """Render the v9 dashboard PNG. Returns the image path."""
    stem = audio_path.stem
    img_path = HERE / f"_{stem}_tmp.png"
    duration = data["duration"]
    metrics = data["metrics"]

    fig, (ax_wave, ax_stats, ax) = plt.subplots(
        3, 1, figsize=(12.8, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 0.42, 3]}, constrained_layout=True,
    )
    fig.set_layout_engine("constrained", h_pad=0, w_pad=0, hspace=0, wspace=0)

    ax_wave.plot(data["wave_x"], data["wave_yT"], color=C_WAVE, alpha=0.7, linewidth=0.5)
    ax_wave.set_ylabel("Amplitude", color=C_WAVE)
    ax_wave.tick_params(axis="y", labelcolor=C_WAVE)
    ax_wave.grid(True, alpha=0.2)

    ax_int = ax_wave.twinx()
    ax_int.plot(data["int_x"], data["int_yT"], color=C_INT, linewidth=1.5, label="Intensity")
    ax_int.set_ylabel("Intensity (dB)", color=C_INT)
    ax_int.tick_params(axis="y", labelcolor=C_INT)
    ax_int.set_ylim(0, 90)

    ax.pcolormesh(data["sg_X"], data["sg_Y"], data["sg_db"], cmap="viridis",
                  vmin=data["vmin"], vmax=data["vmax"], shading="auto")
    ax.set_ylim(0, 5000)
    ax.set_ylabel("Frequency (Hz)")

    ax2 = ax.twinx()
    ax2.plot(data["pitch_x"], data["pitch_y"], "-", color=C_PITCH, linewidth=1.5, label="Pitch")
    ax2.set_ylabel("Pitch (Hz)", color=C_PITCH)
    ax2.tick_params(axis="y", labelcolor=C_PITCH)
    ax2.set_ylim(50, 450)

    if len(data["f_t"]):
        ax.scatter(data["f_t"], data["f1"], c=C_F1, s=6, alpha=0.8, label="F1")
        ax.scatter(data["f_t"], data["f2"], c=C_F2, s=6, alpha=0.8, label="F2")
        ax.scatter(data["f_t"], data["f3"], c=C_F3, s=6, alpha=0.8, label="F3")

    valid_pitches = data["pitch_y"][~np.isnan(data["pitch_y"])]
    try:
        _draw_stats_grid(ax_stats, metrics, data["voiced_ratio"], valid_pitches)
    except Exception:
        ax_stats.axis("off")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines1 or lines2:
        ax_stats.legend(
            handles=lines1 + lines2, labels=labels1 + labels2,
            loc="center right", fontsize=7.5, frameon=False, ncol=2,
            labelcolor="#1a1a1a", borderpad=0.2, labelspacing=0.3,
            handlelength=0.7, handletextpad=0.3, columnspacing=0.8,
        )
    ax.set_xlim(0, duration)
    fig.savefig(str(img_path), dpi=150, format="png")
    plt.close(fig)
    return img_path


def build_video_static(video_path: Path, img_path: Path, audio_path: Path, duration: float):
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-framerate", "1", "-i", str(img_path),
        "-i", str(audio_path),
        "-c:v", "libsvtav1",
        "-svtav1-params", "preset=8:crf=35:scd=0:keyint=1000:tune=0",
        "-pix_fmt", "yuv420p",
        "-c:a", "libopus", "-b:a", "64k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-t", f"{duration:.3f}", "-shortest",
        "-movflags", "+faststart",
        str(video_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True, timeout=300)


# ══════════════════════════════════════════════════════════════════════════
#  SCROLL MODE  (v6 animation, fed v9 metrics)
# ══════════════════════════════════════════════════════════════════════════

FPS = 30                       # output video frame rate (scroll)
WINDOW_SECONDS = 10            # seconds visible at once while scrolling

FRAME_W, FRAME_H = 1920, 1080
SIDEBAR_L = 120
SIDEBAR_R = 120

Y_AMPLITUDE = (-0.5, 0.5)
Y_INTENSITY = (0, 80)
Y_FREQUENCY = (0, 5000)
Y_PITCH = (50, 450)

PLAYHEAD_COLOR = "red@0.85"
PLAYHEAD_W = 3
TILE_MAX_PX = 60000

SVTAV1_PARAMS = "preset=8:crf=35:scd=0:keyint=1000:tune=0"
OPUS_BITRATE = "64k"

CONTENT_X = SIDEBAR_L
CONTENT_W = FRAME_W - SIDEBAR_L - SIDEBAR_R
PANEL_TOP_Y = 70
TOP_H = 250
GAP = 20
BOT_TOP_Y = PANEL_TOP_Y + TOP_H + GAP
BOT_H = 650
XLABEL_H = 40
STRIP_H = TOP_H + GAP + BOT_H + XLABEL_H
STRIP_TOP_BAND = 0
STRIP_BOT_BAND = TOP_H + GAP

YT_AMP = [-0.5, -0.25, 0, 0.25, 0.5]
YT_INT = [0, 20, 40, 60, 80]
YT_FREQ = [0, 1000, 2000, 3000, 4000, 5000]
YT_PITCH = [50, 150, 250, 350, 450]


def stats_line(data: dict) -> str:
    """One-line stats banner — now carries the full v9 metric set."""
    m = data["metrics"]
    valid = data["pitch_y"][~np.isnan(data["pitch_y"])]
    if len(valid):
        mean_p = float(np.mean(valid)); sd_p = float(np.std(valid))
        med_p = float(np.median(valid))
        p5, p95 = (float(x) for x in np.percentile(valid, [5, 95]))
    else:
        mean_p = sd_p = med_p = p5 = p95 = 0.0
    hnr_s = f"{m['hnr']:.0f} dB" if m["hnr"] is not None else "N/A"
    cpp_s = f"{m['cpp']:.1f} dB" if m["cpp"] is not None else "N/A"
    jit_s = f"{m['jitter']:.2f}%" if m["jitter"] is not None else "N/A"
    shi_s = f"{m['shimmer']:.2f}%" if m["shimmer"] is not None else "N/A"
    tilt_s = f"{m['tilt_slope']:.1f} dB/oct" if m["tilt_slope"] is not None else "N/A"
    return (f"Pitch: {mean_p:.0f} Hz   Median: {med_p:.0f} ±{sd_p:.0f}   "
            f"Range: {p5:.0f}-{p95:.0f}   HNR: {hnr_s}   CPP: {cpp_s}   "
            f"Jitter: {jit_s}   Shimmer: {shi_s}   Tilt: {tilt_s}   "
            f"Voiced: {data['voiced_ratio'] * 100:.0f}%")


def _rect(x_px, y_top_px, w_px, h_px, fig_w, fig_h):
    """Pixel rect (top-origin) -> matplotlib add_axes rect (bottom-origin)."""
    return [x_px / fig_w, 1 - (y_top_px + h_px) / fig_h, w_px / fig_w, h_px / fig_h]


def _time_formatter(duration):
    def fmt(x, pos):
        if duration >= 60:
            m = int(x // 60); s = int(x % 60)
            return f"{m}:{s:02d}"
        return f"{x:g}"
    return FuncFormatter(fmt)


def render_strip_tile(out_png: Path, w_px: int, xlim, data: dict, visible_span: float):
    fig = plt.figure(figsize=(w_px / 100.0, STRIP_H / 100.0), dpi=100)
    ax_wave = fig.add_axes(_rect(0, STRIP_TOP_BAND, w_px, TOP_H, w_px, STRIP_H))
    ax_int = ax_wave.twinx()
    ax_spec = fig.add_axes(_rect(0, STRIP_BOT_BAND, w_px, BOT_H, w_px, STRIP_H))
    ax_pitch = ax_spec.twinx()

    ax_wave.plot(data["wave_x"], data["wave_y"], color=C_WAVE, alpha=0.7, linewidth=0.5)
    ax_wave.set_ylim(*Y_AMPLITUDE)
    ax_int.plot(data["int_x"], data["int_y"], color=C_INT, linewidth=1.5)
    ax_int.set_ylim(*Y_INTENSITY)

    ax_spec.pcolormesh(data["sg_X"], data["sg_Y"], data["sg_db"], cmap="viridis",
                       vmin=data["vmin"], vmax=data["vmax"], shading="auto")
    ax_spec.set_ylim(*Y_FREQUENCY)
    if len(data["f_t"]):
        ax_spec.scatter(data["f_t"], data["f1"], c=C_F1, s=6, alpha=0.85)
        ax_spec.scatter(data["f_t"], data["f2"], c=C_F2, s=6, alpha=0.85)
        ax_spec.scatter(data["f_t"], data["f3"], c=C_F3, s=6, alpha=0.85)
    ax_pitch.plot(data["pitch_x"], data["pitch_y"], "-", color=C_PITCH, linewidth=1.5)
    ax_pitch.set_ylim(*Y_PITCH)

    step = 1 if visible_span <= 12 else (2 if visible_span <= 30 else 5)
    for ax in (ax_wave, ax_int, ax_spec, ax_pitch):
        ax.set_xlim(*xlim)
        ax.set_yticks({id(ax_wave): YT_AMP, id(ax_int): YT_INT,
                       id(ax_spec): YT_FREQ, id(ax_pitch): YT_PITCH}[id(ax)])
        ax.tick_params(left=False, right=False, labelleft=False, labelright=False)
        ax.xaxis.set_major_locator(MultipleLocator(step))
        for sp in ax.spines.values():
            sp.set_visible(False)

    ax_wave.tick_params(labelbottom=False)
    ax_spec.tick_params(labelbottom=True, colors="#333333")
    ax_spec.xaxis.set_major_formatter(_time_formatter(data["duration"]))
    ax_wave.grid(True, axis="both", alpha=0.15)

    fig.savefig(str(out_png), dpi=100, facecolor="white")
    plt.close(fig)


def build_strip(strip_png: Path, strip_w: int, pps: float, data: dict, visible_span: float):
    n_tiles = max(1, math.ceil(strip_w / TILE_MAX_PX))
    if n_tiles == 1:
        render_strip_tile(strip_png, strip_w, (0.0, data["duration"]), data, visible_span)
        return 1

    bounds = [round(i * strip_w / n_tiles) for i in range(n_tiles + 1)]
    tile_paths = []
    for i in range(n_tiles):
        x0, x1 = bounds[i], bounds[i + 1]
        tp = strip_png.with_name(f"{strip_png.stem}_tile{i}.png")
        render_strip_tile(tp, x1 - x0, (x0 / pps, x1 / pps), data, visible_span)
        tile_paths.append(tp)

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for tp in tile_paths:
        cmd += ["-i", str(tp)]
    cmd += ["-filter_complex", f"hstack=inputs={n_tiles}", str(strip_png)]
    subprocess.run(cmd, check=True, timeout=600)
    for tp in tile_paths:
        tp.unlink(missing_ok=True)
    return n_tiles


def build_chrome(chrome_png: Path, title: str, data: dict):
    """Static frame: fixed sidebars with Y scales + stats line + legend."""
    fig = plt.figure(figsize=(FRAME_W / 100.0, FRAME_H / 100.0), dpi=100)
    ax_wave = fig.add_axes(_rect(CONTENT_X, PANEL_TOP_Y, CONTENT_W, TOP_H, FRAME_W, FRAME_H))
    ax_int = ax_wave.twinx()
    ax_spec = fig.add_axes(_rect(CONTENT_X, BOT_TOP_Y, CONTENT_W, BOT_H, FRAME_W, FRAME_H))
    ax_pitch = ax_spec.twinx()

    for ax, ylim, yt in ((ax_wave, Y_AMPLITUDE, YT_AMP), (ax_int, Y_INTENSITY, YT_INT),
                         (ax_spec, Y_FREQUENCY, YT_FREQ), (ax_pitch, Y_PITCH, YT_PITCH)):
        ax.set_ylim(*ylim)
        ax.set_yticks(yt)
        ax.set_xticks([])
        ax.patch.set_alpha(0.0)

    ax_wave.set_ylabel("Amplitude", color=C_WAVE)
    ax_wave.tick_params(axis="y", labelcolor=C_WAVE)
    ax_int.set_ylabel("Intensity (dB)", color=C_INT)
    ax_int.tick_params(axis="y", labelcolor=C_INT)
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_pitch.set_ylabel("Pitch (Hz)", color=C_PITCH)
    ax_pitch.tick_params(axis="y", labelcolor=C_PITCH)

    fig.text(0.006, 0.975, title, ha="left", va="top", fontsize=11,
             family="monospace", color="#000000")
    fig.text(0.5, 0.975, stats_line(data), ha="center", va="top", fontsize=8.5,
             family="monospace", color="#000000")

    handles = [
        Line2D([], [], color=C_INT, lw=2, label="Intensity"),
        Line2D([], [], color=C_PITCH, lw=2, label="Pitch"),
        Line2D([], [], marker="o", ls="", color=C_F1, label="F1"),
        Line2D([], [], marker="o", ls="", color=C_F2, label="F2"),
        Line2D([], [], marker="o", ls="", color=C_F3, label="F3"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.996, 0.99),
               handletextpad=0.3, columnspacing=1.0)

    fig.savefig(str(chrome_png), dpi=100, facecolor="white")
    plt.close(fig)


def build_video_scroll(video_path, chrome_png, strip_png, audio_path, strip_w, pps, duration):
    maxoff = strip_w - CONTENT_W
    half = CONTENT_W / 2.0
    crop_x = f"max(0,min({maxoff},t*{pps:.6f}-{half}))"
    play_x = f"{CONTENT_X}+t*{pps:.6f}-({crop_x})"

    fc = (
        f"[1:v]crop={CONTENT_W}:{STRIP_H}:x='{crop_x}':y=0[win];"
        f"[0:v][win]overlay={CONTENT_X}:{PANEL_TOP_Y}[b];"
        f"[b][3:v]overlay=x='{play_x}':y={PANEL_TOP_Y},format=yuv420p[v]"
    )

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-stats",
        "-loop", "1", "-framerate", str(FPS), "-i", str(chrome_png),
        "-loop", "1", "-framerate", str(FPS), "-i", str(strip_png),
        "-i", str(audio_path),
        "-f", "lavfi", "-i", f"color=c={PLAYHEAD_COLOR}:s={PLAYHEAD_W}x{STRIP_H}:r={FPS}",
        "-filter_complex", fc,
        "-map", "[v]", "-map", "2:a:0",
        "-t", f"{duration:.3f}", "-r", str(FPS),
        "-c:v", "libsvtav1", "-svtav1-params", SVTAV1_PARAMS,
        "-pix_fmt", "yuv420p",
        "-c:a", "libopus", "-b:a", OPUS_BITRATE,
        "-movflags", "+faststart",
        str(video_path),
    ]
    subprocess.run(cmd, check=True, timeout=1800)


# ── per-file processing ─────────────────────────────────────────────────────

def process_one(audio_path: Path, scroll: bool, window: int, fps: int):
    stem = audio_path.stem
    if audio_path.suffix.lower() not in AUDIO_EXTS or audio_path.name.startswith("_"):
        return

    video_path = HERE / f"{stem}.mp4"
    if video_path.exists():
        print(f"  SKIP {stem}.mp4 — already exists")
        return

    print(f"  {stem}")
    wav_path, is_temp_wav = ensure_wav(audio_path)

    sound = parselmouth.Sound(str(wav_path))
    data = analyse(sound)
    duration = data["duration"]

    if not scroll:
        # ── static (v9) ──
        img_path = render_static(audio_path, data)
        build_video_static(video_path, img_path, audio_path, duration)
        img_path.unlink(missing_ok=True)
    else:
        # ── scroll (v6, v9 metrics) ──
        global FPS, WINDOW_SECONDS
        FPS = fps
        WINDOW_SECONDS = window
        if duration <= window:
            pps = CONTENT_W / duration
            visible_span = duration
            mode = "single"
        else:
            pps = CONTENT_W / window
            visible_span = window
            mode = "scroll"
        strip_w = round(duration * pps)

        chrome_png = HERE / f"_{stem}_chrome.png"
        strip_png = HERE / f"_{stem}_strip.png"
        print(f"    {duration:.1f}s {mode} strip={strip_w}px @ {fps}fps")

        build_chrome(chrome_png, audio_path.name, data)
        n_tiles = build_strip(strip_png, strip_w, pps, data, visible_span)
        build_video_scroll(video_path, chrome_png, strip_png, audio_path, strip_w, pps, duration)

        chrome_png.unlink(missing_ok=True)
        strip_png.unlink(missing_ok=True)

    if is_temp_wav and wav_path.exists():
        wav_path.unlink()

    size_mb = video_path.stat().st_size / 1024 / 1024
    print(f"    -> {video_path.name}  ({duration:.1f}s, {size_mb:.1f} MB)")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Voice-profile videos (v10).")
    ap.add_argument("--scroll", action="store_true",
                    help="v6-style scrolling animation (slow). Default: static v9 dashboard.")
    ap.add_argument("--window", type=int, default=WINDOW_SECONDS,
                    help=f"seconds visible while scrolling (scroll only, default {WINDOW_SECONDS})")
    ap.add_argument("--fps", type=int, default=FPS,
                    help=f"output frame rate (scroll only, default {FPS})")
    ap.add_argument("files", nargs="*", type=Path,
                    help="specific audio files (default: every audio file in this folder)")
    args = ap.parse_args()

    if args.files:
        audio_files = sorted(f if f.is_absolute() else (HERE / f) for f in args.files)
    else:
        by_stem: dict[str, Path] = {}
        for p in sorted(HERE.iterdir()):
            if p.suffix.lower() not in AUDIO_EXTS or p.name.startswith("_"):
                continue
            stem = p.stem
            if stem in by_stem:
                existing = by_stem[stem]
                if p.suffix.lower() == ".wav" and existing.suffix.lower() != ".wav":
                    continue
            by_stem[stem] = p
        audio_files = sorted(by_stem.values())

    if not audio_files:
        print("No audio files found in", HERE)
        sys.exit(1)

    mode = "scroll" if args.scroll else "static"
    print(f"Found {len(audio_files)} audio file(s) in {HERE}  [{mode} mode]\n")

    t0 = time.time()
    for af in audio_files:
        try:
            process_one(af, args.scroll, args.window, args.fps)
        except subprocess.CalledProcessError as e:
            print(f"  !! ffmpeg failed for {af.name} (rc={e.returncode})")

    print(f"\nDone in {time.time() - t0:.0f}s  |  {len(audio_files)} file(s)")


if __name__ == "__main__":
    main()
