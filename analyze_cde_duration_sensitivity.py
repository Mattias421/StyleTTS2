#!/usr/bin/env python3
"""Measure how CDE text representations change when token durations change."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

from models import build_model, load_ASR_models, load_F0_models
from text_to_speech import load_inference_checkpoint
from text_utils import TextCleaner, symbols
from Utils.PLBERT.util import load_plbert
from utils import mask_from_lens, maximum_path, recursive_munch


MEAN = -4.0
STD = 4.0
CONTROL_NAMES = ("scale_0.5", "scale_2.0", "equal", "shuffle", "reverse")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Hold text-encoder outputs fixed and measure their CDE sensitivity "
            "to aligned and predicted duration variants."
        )
    )
    parser.add_argument(
        "-p",
        "--config-path",
        type=Path,
        default=Path("Configs/cde_duration_sensitivity.yml"),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run lightweight tests that do not require checkpoints or audio.",
    )
    return parser.parse_args()


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(config):
    configured = str(config.get("device", "auto")).lower()
    if configured in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if configured.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(configured)


def length_to_mask(lengths: torch.Tensor):
    positions = torch.arange(lengths.max(), device=lengths.device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)


def parse_esd_list(path: Path):
    entries = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for row, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                raise ValueError(f"{path}:{row} must contain wav|text|speaker_id")
            entries.append(
                {
                    "row": row,
                    "relative_wav": parts[0],
                    "phonemized_text": parts[1],
                    "speaker_id": parts[2],
                }
            )
    return entries


def choose_entries(entries, sample_count: int, detail_rows, seed: int):
    detail_rows = {int(row) for row in detail_rows}
    by_row = {entry["row"]: entry for entry in entries}
    missing = sorted(detail_rows - by_row.keys())
    if missing:
        raise ValueError(f"Detailed rows are absent from the text list: {missing}")

    count = min(int(sample_count), len(entries))
    rng = np.random.default_rng(seed)
    chosen_indices = set(rng.choice(len(entries), size=count, replace=False).tolist())
    chosen_indices.update(entries.index(by_row[row]) for row in detail_rows)
    chosen = [entries[index] for index in sorted(chosen_indices)]
    return chosen, detail_rows


def build_mel_transform(model_config):
    preprocess = model_config.get("preprocess_params", {})
    spect = preprocess.get("spect_params", {})
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=int(preprocess.get("sr", 24000)),
        n_mels=int(model_config["model_params"].get("n_mels", 80)),
        n_fft=int(spect.get("n_fft", 2048)),
        win_length=int(spect.get("win_length", 1200)),
        hop_length=int(spect.get("hop_length", 300)),
    )


def load_sample(entry, root_path, sample_rate, mel_transform, text_cleaner, device):
    wav_path = root_path / entry["relative_wav"]
    wave, source_rate = sf.read(wav_path, dtype="float32")
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if source_rate != sample_rate:
        wave = librosa.resample(
            wave, orig_sr=source_rate, target_sr=sample_rate
        ).astype(np.float32)

    # Match FilePathDataset: alignment training sees 5000 samples of padding.
    wave = np.pad(wave, (5000, 5000))
    wave_tensor = torch.from_numpy(wave).float()
    mel = mel_transform(wave_tensor)
    mel = (torch.log(1e-5 + mel.unsqueeze(0)) - MEAN) / STD
    mel = mel.squeeze(0)
    mel = mel[:, : mel.shape[-1] - mel.shape[-1] % 2]

    token_ids = text_cleaner(entry["phonemized_text"])
    token_ids.insert(0, 0)
    token_ids.append(0)
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    input_lengths = torch.tensor([tokens.shape[-1]], dtype=torch.long, device=device)
    mels = mel.to(device).unsqueeze(0)
    mel_lengths = torch.tensor([mel.shape[-1]], dtype=torch.long, device=device)
    return tokens, input_lengths, mels, mel_lengths


def load_analysis_model(model_config_path: Path, checkpoint_path: Path, device):
    model_config = load_yaml(model_config_path)
    text_aligner = load_ASR_models(
        model_config.get("ASR_path", False),
        model_config.get("ASR_config", False),
    )
    pitch_extractor = load_F0_models(model_config.get("F0_path", False))
    plbert = load_plbert(model_config.get("PLBERT_dir", False))
    model_params = recursive_munch(model_config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    if "cde" not in model:
        raise ValueError(f"{model_config_path} does not enable model.cde")

    state = load_inference_checkpoint(model, checkpoint_path)
    if "cde" not in state.get("net", {}):
        raise ValueError(f"{checkpoint_path} does not contain CDE parameters")
    for module in model.values():
        module.eval()
        module.to(device)
    return model, model_config


def aligned_durations(model, tokens, input_lengths, mels, mel_lengths):
    n_down = int(model.text_aligner.n_down)
    down_lengths = mel_lengths // (2**n_down)
    mel_mask = length_to_mask(down_lengths)
    _, _, attention = model.text_aligner(mels, mel_mask, tokens)
    attention = attention.transpose(-1, -2)
    attention = attention[..., 1:]
    attention = attention.transpose(-1, -2)
    path_mask = mask_from_lens(attention, input_lengths, down_lengths)
    monotonic_attention = maximum_path(attention, path_mask)
    return monotonic_attention.sum(axis=-1).detach()


def predicted_durations(model, tokens, input_lengths, text_mask, mels):
    bert_duration = model.bert(tokens, attention_mask=(~text_mask).int())
    duration_encoding = model.bert_encoder(bert_duration).transpose(-1, -2)
    style = model.predictor_encoder(mels.unsqueeze(1))
    encoded = model.predictor.text_encoder(
        duration_encoding, style, input_lengths, text_mask
    )
    predicted, _ = model.predictor.lstm(encoded)
    logits = model.predictor.duration_proj(predicted)
    frame_durations = torch.sigmoid(logits).sum(axis=-1)
    frame_durations = torch.round(frame_durations).clamp(min=1)
    n_down = int(model.text_aligner.n_down)
    return (frame_durations / (2**n_down)).clamp(min=1).detach()


def normalized_phone_starts(durations: np.ndarray):
    durations = np.maximum(np.asarray(durations, dtype=np.float64), 1e-3)
    starts = np.concatenate(([0.0], np.cumsum(durations[:-1])))
    return starts / max(float(durations.sum()), 1.0)


def build_duration_variants(
    baseline: torch.Tensor,
    valid_length: int,
    log_epsilon: float,
    seed: int,
):
    baseline = baseline[:valid_length].float().clone()
    interior = list(range(1, max(1, valid_length - 1)))
    variants = {"baseline": baseline}
    variants["scale_0.5"] = baseline * 0.5
    variants["scale_2.0"] = baseline * 2.0

    equal = baseline.clone()
    if interior:
        equal[interior] = baseline[interior].sum() / len(interior)
    variants["equal"] = equal

    rng = np.random.default_rng(seed)
    shuffled = baseline.clone()
    if interior:
        permutation = torch.as_tensor(
            rng.permutation(len(interior)), device=baseline.device
        )
        shuffled[interior] = baseline[interior][permutation]
    variants["shuffle"] = shuffled

    reversed_duration = baseline.clone()
    if interior:
        reversed_duration[interior] = baseline[interior].flip(0)
    variants["reverse"] = reversed_duration

    factor = math.exp(float(log_epsilon))
    for token_index in interior:
        plus = baseline.clone()
        minus = baseline.clone()
        plus[token_index] *= factor
        minus[token_index] /= factor
        variants[f"plus_{token_index}"] = plus
        variants[f"minus_{token_index}"] = minus
    return variants, interior


def run_cde_variants(model, t_en, mask, variants, batch_size):
    names = list(variants)
    outputs = {}
    for offset in range(0, len(names), batch_size):
        chunk_names = names[offset : offset + batch_size]
        duration_batch = torch.stack([variants[name] for name in chunk_names])
        input_batch = t_en.expand(len(chunk_names), -1, -1)
        mask_batch = mask.expand(len(chunk_names), -1, -1)
        result = model.cde(input_batch, mask_batch, duration_batch)
        for index, name in enumerate(chunk_names):
            outputs[name] = result[index].detach().float().cpu().numpy()
    return outputs


def representation_metrics(baseline, variant, token_indices):
    baseline = baseline[:, token_indices].astype(np.float64, copy=False)
    variant = variant[:, token_indices].astype(np.float64, copy=False)
    delta = variant - baseline
    token_l2 = np.linalg.norm(delta, axis=0)
    baseline_norm = np.linalg.norm(baseline)
    flat_baseline = baseline.reshape(-1)
    flat_variant = variant.reshape(-1)
    cosine_denom = np.linalg.norm(flat_baseline) * np.linalg.norm(flat_variant)
    cosine = (
        float(np.dot(flat_baseline, flat_variant) / cosine_denom)
        if cosine_denom > 0
        else 1.0
    )
    max_position = int(np.argmax(token_l2)) if token_l2.size else -1
    return {
        "frobenius": float(np.linalg.norm(delta)),
        "relative_frobenius": float(np.linalg.norm(delta) / max(baseline_norm, 1e-12)),
        "mean_token_l2": float(token_l2.mean()) if token_l2.size else 0.0,
        "cosine_distance": float(1.0 - cosine),
        "max_token_position": max_position,
        "max_token_l2": float(token_l2[max_position]) if token_l2.size else 0.0,
        "max_abs": float(np.abs(delta).max()) if delta.size else 0.0,
    }


def sensitivity_matrix(outputs, interior, log_epsilon):
    rows = []
    denominator = 2.0 * float(log_epsilon)
    for token_index in interior:
        derivative = (
            outputs[f"plus_{token_index}"] - outputs[f"minus_{token_index}"]
        ) / denominator
        rows.append(np.linalg.norm(derivative[:, interior], axis=0))
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


def sensitivity_metrics(matrix):
    if matrix.size == 0:
        return {
            "sensitivity_mean": 0.0,
            "sensitivity_max": 0.0,
            "sensitivity_diagonal_mean": 0.0,
            "sensitivity_off_diagonal_mean": 0.0,
            "sensitivity_forward_fraction": 0.0,
            "sensitivity_backward_fraction": 0.0,
        }
    diagonal = np.diag(matrix)
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    total = float(matrix.sum())
    forward = float(np.triu(matrix, k=1).sum())
    backward = float(np.tril(matrix, k=-1).sum())
    return {
        "sensitivity_mean": float(matrix.mean()),
        "sensitivity_max": float(matrix.max()),
        "sensitivity_diagonal_mean": float(diagonal.mean()),
        "sensitivity_off_diagonal_mean": (
            float(off_diagonal.mean()) if off_diagonal.size else 0.0
        ),
        "sensitivity_forward_fraction": forward / max(total, 1e-12),
        "sensitivity_backward_fraction": backward / max(total, 1e-12),
    }


def token_labels(token_ids):
    labels = []
    for index, token_id in enumerate(token_ids):
        symbol = symbols[int(token_id)] if int(token_id) < len(symbols) else "?"
        labels.append(f"{index}:{symbol}")
    return labels


def safe_stem(entry):
    return Path(entry["relative_wav"]).with_suffix("").as_posix().replace("/", "_")


def write_detail_plot(
    output_base,
    title,
    labels,
    baseline_duration,
    sensitivity,
    metric_rows,
):
    interior_labels = labels[1:-1]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(max(10, len(labels) * 0.45), 12),
        constrained_layout=True,
    )
    axes[0].bar(np.arange(len(labels)), baseline_duration)
    axes[0].set_title(f"{title}\nBaseline durations")
    axes[0].set_ylabel("CDE duration")
    axes[0].set_xticks(np.arange(len(labels)), labels, rotation=90)

    image = axes[1].imshow(sensitivity, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("Local log-duration sensitivity")
    axes[1].set_xlabel("Affected output token")
    axes[1].set_ylabel("Perturbed duration token")
    axes[1].set_xticks(np.arange(len(interior_labels)), interior_labels, rotation=90)
    axes[1].set_yticks(np.arange(len(interior_labels)), interior_labels)
    fig.colorbar(image, ax=axes[1], label="Channel L2 derivative")

    control_rows = [
        row
        for row in metric_rows
        if row["representation"] == "output" and row["variant"] in CONTROL_NAMES
    ]
    axes[2].bar(
        np.arange(len(control_rows)),
        [row["relative_frobenius"] for row in control_rows],
    )
    axes[2].set_title("Control perturbation distances")
    axes[2].set_ylabel("Relative Frobenius distance")
    axes[2].set_xticks(
        np.arange(len(control_rows)),
        [row["variant"] for row in control_rows],
        rotation=25,
    )

    fig.savefig(output_base.with_suffix(".png"), dpi=160)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_metric_rows(rows):
    grouped = defaultdict(list)
    numeric_keys = (
        "frobenius",
        "relative_frobenius",
        "mean_token_l2",
        "cosine_distance",
        "max_token_l2",
        "max_abs",
    )
    for row in rows:
        key = (
            row["model"],
            row["duration_source"],
            row["representation"],
            row["variant"],
        )
        grouped[key].append(row)

    summaries = []
    for key, group in sorted(grouped.items()):
        summary = {
            "model": key[0],
            "duration_source": key[1],
            "representation": key[2],
            "variant": key[3],
            "samples": len(group),
        }
        for metric in numeric_keys:
            values = np.asarray([float(row[metric]) for row in group])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std())
        summaries.append(summary)
    return summaries


def aggregate_sensitivity_rows(rows):
    grouped = defaultdict(list)
    numeric_keys = (
        "sensitivity_mean",
        "sensitivity_max",
        "sensitivity_diagonal_mean",
        "sensitivity_off_diagonal_mean",
        "sensitivity_forward_fraction",
        "sensitivity_backward_fraction",
    )
    for row in rows:
        grouped[(row["model"], row["duration_source"])].append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        summary = {
            "model": key[0],
            "duration_source": key[1],
            "samples": len(group),
        }
        for metric in numeric_keys:
            values = np.asarray([float(row[metric]) for row in group])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std())
        summaries.append(summary)
    return summaries


def write_aggregate_plots(output_dir, metric_summaries, sensitivity_summaries):
    if metric_summaries:
        control_rows = [
            row for row in metric_summaries if row["representation"] == "output"
        ]
        groups = sorted(
            {(row["model"], row["duration_source"]) for row in control_rows}
        )
        variants = list(CONTROL_NAMES)
        x = np.arange(len(groups))
        width = 0.8 / len(variants)
        fig, ax = plt.subplots(figsize=(max(11, len(groups) * 1.4), 6))
        for variant_index, variant in enumerate(variants):
            values = []
            errors = []
            for model_name, source_name in groups:
                match = next(
                    row
                    for row in control_rows
                    if row["model"] == model_name
                    and row["duration_source"] == source_name
                    and row["variant"] == variant
                )
                values.append(match["relative_frobenius_mean"])
                errors.append(match["relative_frobenius_std"])
            positions = x - 0.4 + width / 2 + variant_index * width
            ax.bar(
                positions,
                values,
                width,
                yerr=errors,
                capsize=2,
                label=variant,
            )
        ax.set_title("CDE output sensitivity to duration controls")
        ax.set_ylabel("Mean relative Frobenius distance")
        ax.set_xticks(
            x,
            [f"{model}\n{source}" for model, source in groups],
            rotation=30,
            ha="right",
        )
        ax.legend(ncols=min(5, len(variants)))
        fig.tight_layout()
        fig.savefig(output_dir / "aggregate_control_distances.png", dpi=160)
        fig.savefig(output_dir / "aggregate_control_distances.pdf", bbox_inches="tight")
        plt.close(fig)

    if sensitivity_summaries:
        groups = [
            (row["model"], row["duration_source"]) for row in sensitivity_summaries
        ]
        x = np.arange(len(groups))
        width = 0.38
        diagonal = [
            row["sensitivity_diagonal_mean_mean"] for row in sensitivity_summaries
        ]
        off_diagonal = [
            row["sensitivity_off_diagonal_mean_mean"] for row in sensitivity_summaries
        ]
        fig, ax = plt.subplots(figsize=(max(11, len(groups) * 1.4), 6))
        ax.bar(x - width / 2, diagonal, width, label="Diagonal")
        ax.bar(x + width / 2, off_diagonal, width, label="Off diagonal")
        ax.set_title("Local duration sensitivity by model")
        ax.set_ylabel("Mean channel L2 derivative")
        ax.set_xticks(
            x,
            [f"{model}\n{source}" for model, source in groups],
            rotation=30,
            ha="right",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "aggregate_sensitivity.png", dpi=160)
        fig.savefig(output_dir / "aggregate_sensitivity.pdf", bbox_inches="tight")
        plt.close(fig)


def analyze_duration_source(
    model,
    model_name,
    entry,
    source_name,
    durations,
    t_en,
    cde_mask,
    token_ids,
    output_dir,
    log_epsilon,
    variant_batch_size,
    seed,
    detailed,
):
    valid_length = len(token_ids)
    variants, interior = build_duration_variants(
        durations.squeeze(0), valid_length, log_epsilon, seed
    )
    outputs = run_cde_variants(model, t_en, cde_mask, variants, variant_batch_size)
    baseline_output = outputs["baseline"]
    fixed_input = t_en[0].detach().float().cpu().numpy()
    baseline_residual = baseline_output - fixed_input
    metric_rows = []
    for variant_name in CONTROL_NAMES:
        for representation, baseline, variant in (
            ("output", baseline_output, outputs[variant_name]),
            (
                "residual",
                baseline_residual,
                outputs[variant_name] - fixed_input,
            ),
        ):
            metrics = representation_metrics(baseline, variant, interior)
            metric_rows.append(
                {
                    "model": model_name,
                    "row": entry["row"],
                    "source_path": entry["relative_wav"],
                    "duration_source": source_name,
                    "representation": representation,
                    "variant": variant_name,
                    **metrics,
                }
            )

    sensitivity = sensitivity_matrix(outputs, interior, log_epsilon)
    sensitivity_row = {
        "model": model_name,
        "row": entry["row"],
        "source_path": entry["relative_wav"],
        "duration_source": source_name,
        **sensitivity_metrics(sensitivity),
    }

    labels = token_labels(token_ids)
    sample_dir = output_dir / model_name / f"row_{entry['row']:04d}_{safe_stem(entry)}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    control_outputs = np.stack([outputs[name] for name in CONTROL_NAMES])
    control_durations = np.stack(
        [variants[name].detach().cpu().numpy() for name in CONTROL_NAMES]
    )
    np.savez_compressed(
        sample_dir / f"{source_name}.npz",
        token_ids=np.asarray(token_ids, dtype=np.int64),
        token_labels=np.asarray(labels),
        fixed_t_en=fixed_input,
        baseline_durations=variants["baseline"].detach().cpu().numpy(),
        baseline_output=baseline_output,
        baseline_residual=baseline_residual,
        control_names=np.asarray(CONTROL_NAMES),
        control_durations=control_durations,
        control_outputs=control_outputs,
        sensitivity=sensitivity,
        interior_token_indices=np.asarray(interior, dtype=np.int64),
        log_epsilon=np.asarray(log_epsilon),
    )

    if detailed:
        title = f"{model_name} | row {entry['row']} | {source_name}"
        write_detail_plot(
            sample_dir / source_name,
            title,
            labels,
            variants["baseline"].detach().cpu().numpy(),
            sensitivity,
            metric_rows,
        )
    return metric_rows, sensitivity_row, sample_dir / f"{source_name}.npz"


def run_analysis(config):
    seed = int(config.get("seed", 0))
    set_seed(seed)
    device = resolve_device(config)
    output_dir = Path(config.get("output_dir", "comparisons/cde_duration_sensitivity"))
    output_dir.mkdir(parents=True, exist_ok=True)
    text_list = Path(config.get("text_list", "Data/val_list_esd.txt"))
    root_path = Path(config["root_path"])
    entries = parse_esd_list(text_list)
    chosen_entries, detail_rows = choose_entries(
        entries,
        int(config.get("sample_count", 100)),
        config.get("detail_rows", [1, 2, 3, 4, 5]),
        seed,
    )
    duration_sources = tuple(config.get("duration_sources", ["aligned", "predicted"]))
    unknown_sources = set(duration_sources) - {"aligned", "predicted"}
    if unknown_sources:
        raise ValueError(f"Unknown duration sources: {sorted(unknown_sources)}")

    log_epsilon = float(config.get("log_duration_epsilon", 0.1))
    if log_epsilon <= 0:
        raise ValueError("log_duration_epsilon must be positive")
    variant_batch_size = int(config.get("variant_batch_size", 16))
    skip_missing = bool(config.get("skip_missing_checkpoints", True))
    text_cleaner = TextCleaner()
    all_metric_rows = []
    all_sensitivity_rows = []
    manifest_rows = []

    for model_index, model_spec in enumerate(config["models"]):
        model_name = str(model_spec["name"])
        model_config_path = Path(model_spec["model_config"])
        checkpoint_path = Path(model_spec["checkpoint"])
        missing_paths = [
            path for path in (model_config_path, checkpoint_path) if not path.exists()
        ]
        if missing_paths and skip_missing:
            print(
                f"Skipping {model_name}; missing: {', '.join(map(str, missing_paths))}"
            )
            continue
        if missing_paths:
            raise FileNotFoundError(", ".join(map(str, missing_paths)))

        print(f"Loading {model_name}: {checkpoint_path}")
        model, model_config = load_analysis_model(
            model_config_path, checkpoint_path, device
        )
        sample_rate = int(model_config.get("preprocess_params", {}).get("sr", 24000))
        mel_transform = build_mel_transform(model_config)

        with torch.inference_mode():
            for sample_index, entry in enumerate(chosen_entries, start=1):
                tokens, input_lengths, mels, mel_lengths = load_sample(
                    entry,
                    root_path,
                    sample_rate,
                    mel_transform,
                    text_cleaner,
                    device,
                )
                text_mask = length_to_mask(input_lengths)
                cde_mask = (~text_mask).unsqueeze(1).float()
                fixed_t_en = model.text_encoder(tokens, input_lengths, text_mask)
                fixed_copy = fixed_t_en.detach().clone()
                source_durations = {}
                if "aligned" in duration_sources:
                    source_durations["aligned"] = aligned_durations(
                        model, tokens, input_lengths, mels, mel_lengths
                    )
                if "predicted" in duration_sources:
                    source_durations["predicted"] = predicted_durations(
                        model, tokens, input_lengths, text_mask, mels
                    )

                token_ids = tokens[0, : input_lengths.item()].detach().cpu().tolist()
                for source_offset, (source_name, durations) in enumerate(
                    source_durations.items()
                ):
                    perturb_seed = (
                        seed
                        + model_index * 1_000_000
                        + entry["row"] * 10
                        + source_offset
                    )
                    metric_rows, sensitivity_row, artifact = analyze_duration_source(
                        model=model,
                        model_name=model_name,
                        entry=entry,
                        source_name=source_name,
                        durations=durations,
                        t_en=fixed_t_en,
                        cde_mask=cde_mask,
                        token_ids=token_ids,
                        output_dir=output_dir,
                        log_epsilon=log_epsilon,
                        variant_batch_size=variant_batch_size,
                        seed=perturb_seed,
                        detailed=entry["row"] in detail_rows,
                    )
                    all_metric_rows.extend(metric_rows)
                    all_sensitivity_rows.append(sensitivity_row)
                    manifest_rows.append(
                        {
                            "model": model_name,
                            "model_config": str(model_config_path),
                            "checkpoint": str(checkpoint_path),
                            "row": entry["row"],
                            "source_path": entry["relative_wav"],
                            "phonemized_text": entry["phonemized_text"],
                            "speaker_id": entry["speaker_id"],
                            "duration_source": source_name,
                            "artifact": str(artifact),
                            "seed": perturb_seed,
                            "log_duration_epsilon": log_epsilon,
                        }
                    )
                if not torch.equal(fixed_t_en, fixed_copy):
                    raise RuntimeError(
                        "The cached t_en changed during perturbation analysis"
                    )
                print(
                    f"[{model_name} {sample_index}/{len(chosen_entries)}] "
                    f"row {entry['row']}: {entry['relative_wav']}"
                )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metric_summaries = aggregate_metric_rows(all_metric_rows)
    sensitivity_summaries = aggregate_sensitivity_rows(all_sensitivity_rows)
    write_csv(output_dir / "metrics.csv", all_metric_rows)
    write_csv(output_dir / "sensitivity.csv", all_sensitivity_rows)
    write_csv(output_dir / "metrics_summary.csv", metric_summaries)
    write_csv(
        output_dir / "sensitivity_summary.csv",
        sensitivity_summaries,
    )
    write_aggregate_plots(output_dir, metric_summaries, sensitivity_summaries)
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "resolved_config.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print(f"Wrote duration-sensitivity analysis to {output_dir}")


def run_self_test():
    baseline = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0])
    variants, interior = build_duration_variants(baseline, 5, 0.1, seed=3)
    assert interior == [1, 2, 3]
    assert set(CONTROL_NAMES).issubset(variants)
    assert torch.isclose(variants["equal"][interior].sum(), baseline[interior].sum())
    assert np.allclose(
        normalized_phone_starts(baseline.numpy()),
        normalized_phone_starts((baseline * 2).numpy()),
        atol=1e-12,
    )

    outputs = {}
    for name, durations in variants.items():
        values = durations.numpy()
        outputs[name] = np.stack((values, values**2))
    matrix = sensitivity_matrix(outputs, interior, 0.1)
    assert matrix.shape == (3, 3)
    metrics = representation_metrics(
        outputs["baseline"], outputs["scale_2.0"], interior
    )
    assert metrics["frobenius"] > 0
    print("Self-test passed.")


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    run_analysis(load_yaml(args.config_path))


if __name__ == "__main__":
    main()
