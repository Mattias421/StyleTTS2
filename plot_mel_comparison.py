#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyworld as pw
import soundfile as sf
import torch
import torchaudio


SAMPLE_RATE = 24000
N_MELS = 80
N_FFT = 2048
WIN_LENGTH = 1200
HOP_LENGTH = 300
MEL_F_MAX = 8000
F0_SHIFT = 256
F0_MIN = 40
F0_MAX = 800
F0_COLOR = "cyan"


def load_features(path):
    wave, sample_rate = sf.read(path, dtype="float32")
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        wave = torchaudio.functional.resample(
            torch.from_numpy(wave), sample_rate, SAMPLE_RATE
        ).numpy()

    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_max=MEL_F_MAX,
    )
    mel = transform(torch.from_numpy(wave)).numpy()
    f0, f0_times = pw.harvest(
        wave.astype(np.float64),
        SAMPLE_RATE,
        f0_floor=F0_MIN,
        f0_ceil=F0_MAX,
        frame_period=F0_SHIFT / SAMPLE_RATE * 1000,
    )
    log_f0 = np.full(f0.shape, np.nan, dtype=np.float64)
    voiced = f0 > 0
    log_f0[voiced] = np.log(f0[voiced])
    return (
        10.0 * np.log10(np.maximum(mel, 1e-10)),
        f0_times,
        log_f0,
        len(wave) / SAMPLE_RATE,
    )


def plot_single(
    mel, f0_times, log_f0, duration, title, output, vmin, vmax, f0_min, f0_max
):
    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(
        mel,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, duration, 0, N_MELS - 1],
        vmin=vmin,
        vmax=vmax,
    )
    ax.set(title=title, xlabel="Time (seconds)", ylabel="Mel bin")
    ax_f0 = ax.twinx()
    ax_f0.plot(
        f0_times, log_f0, color=F0_COLOR, linewidth=1.5, label="Log F0"
    )
    ax_f0.set_ylim(f0_min, f0_max)
    ax_f0.set_ylabel("Log F0 (ln Hz)", color=F0_COLOR)
    ax_f0.tick_params(axis="y", colors=F0_COLOR)
    ax_f0.legend(loc="upper right")
    fig.colorbar(image, ax=ax, label="Log-mel energy (dB)")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h256_l6_wav", type=Path)
    parser.add_argument("esd_small_wav", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    h256_mel, h256_f0_times, h256_log_f0, h256_duration = load_features(
        args.h256_l6_wav
    )
    baseline_mel, baseline_f0_times, baseline_log_f0, baseline_duration = (
        load_features(args.esd_small_wav)
    )

    vmin = min(np.percentile(h256_mel, 1), np.percentile(baseline_mel, 1))
    vmax = max(np.percentile(h256_mel, 99), np.percentile(baseline_mel, 99))
    voiced_log_f0 = np.concatenate(
        (h256_log_f0[~np.isnan(h256_log_f0)], baseline_log_f0[~np.isnan(baseline_log_f0)])
    )
    f0_min = voiced_log_f0.min() - 0.1
    f0_max = voiced_log_f0.max() + 0.1

    plot_single(
        h256_mel,
        h256_f0_times,
        h256_log_f0,
        h256_duration,
        "CDE dt=1.0, h256, l6",
        args.output_dir / "dt100_h256_l6_mel.png",
        vmin,
        vmax,
        f0_min,
        f0_max,
    )
    plot_single(
        baseline_mel,
        baseline_f0_times,
        baseline_log_f0,
        baseline_duration,
        "ESD small baseline",
        args.output_dir / "esd_small_mel.png",
        vmin,
        vmax,
        f0_min,
        f0_max,
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True)
    max_duration = max(h256_duration, baseline_duration)
    f0_axes = []
    for ax, mel, f0_times, log_f0, duration, title in (
        (
            axes[0],
            h256_mel,
            h256_f0_times,
            h256_log_f0,
            h256_duration,
            "CDE dt=1.0, h256, l6",
        ),
        (
            axes[1],
            baseline_mel,
            baseline_f0_times,
            baseline_log_f0,
            baseline_duration,
            "ESD small baseline",
        ),
    ):
        image = ax.imshow(
            mel,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[0, duration, 0, N_MELS - 1],
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_ylabel("Mel bin")
        ax.set_xlim(0, max_duration)
        ax_f0 = ax.twinx()
        ax_f0.plot(
            f0_times, log_f0, color=F0_COLOR, linewidth=1.5, label="Log F0"
        )
        ax_f0.set_ylim(f0_min, f0_max)
        ax_f0.set_ylabel("Log F0 (ln Hz)", color=F0_COLOR)
        ax_f0.tick_params(axis="y", colors=F0_COLOR)
        f0_axes.append(ax_f0)
    f0_axes[0].legend(loc="upper right")
    axes[1].set_xlabel("Time (seconds)")
    fig.colorbar(image, ax=axes, label="Log-mel energy (dB)", pad=0.02)
    fig.savefig(
        args.output_dir / "dt100_h256_l6_vs_esd_small_mels.png",
        dpi=150,
        bbox_inches="tight",
    )
    fig.savefig(
        args.output_dir / "dt100_h256_l6_vs_esd_small_mels.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"h256_l6: {h256_mel.shape[1]} frames, {h256_duration:.3f} seconds")
    print(
        f"esd_small: {baseline_mel.shape[1]} frames, "
        f"{baseline_duration:.3f} seconds"
    )


if __name__ == "__main__":
    main()
