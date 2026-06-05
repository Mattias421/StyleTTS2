#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is already a project dependency.
    yaml = None


REPO_ROOT = Path(__file__).resolve().parent
EVAL_DIR = REPO_ROOT / "Utils" / "Eval"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build mirrored ground-truth samples and run TTS eval scripts."
    )
    parser.add_argument("samples_path", type=Path, help="Directory containing generated wav samples.")
    parser.add_argument(
        "--root-path",
        type=Path,
        default=None,
        help="Dataset root used to resolve relative source wav paths from the manifest.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split name. Inferred from manifest or filenames when omitted.",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=None,
        help="Override output dir for mirrored ground-truth wavs.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy ground-truth wavs instead of symlinking them.",
    )
    parser.add_argument(
        "--refresh-ground-truth",
        action="store_true",
        help="Replace existing mirrored ground-truth wav links/files.",
    )
    parser.add_argument("--nj", type=int, default=16, help="Parallel jobs for F0 evaluation.")
    parser.add_argument(
        "--skip-mcd",
        action="store_true",
        help="Skip mel-cepstral distortion evaluation.",
    )
    parser.add_argument(
        "--skip-f0",
        action="store_true",
        help="Skip log-F0 RMSE evaluation.",
    )
    return parser.parse_args()


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return records


def esd_safe_stem_to_path(stem):
    parts = stem.split("_")
    if len(parts) >= 5 and parts[0] == "ESD":
        return Path(parts[0]) / parts[1] / parts[2] / ("_".join(parts[3:]) + ".wav")
    raise ValueError(
        "Could not infer source wav path from filename. Provide a manifest.jsonl or "
        "use sample names containing __<split>_ESD_<speaker>_<emotion>_<wav-stem>__."
    )


def infer_from_sample_name(path):
    name = path.name
    if "__ref_" not in name:
        raise ValueError(f"Cannot infer source path from sample filename: {name}")
    for marker in ("__val_", "__test_", "__train_"):
        if marker in name:
            split = marker.strip("_")
            source_stem = name.split(marker, 1)[1].split("__ref_", 1)[0]
            return {"split": split, "source_path": esd_safe_stem_to_path(source_stem)}
    raise ValueError(f"Cannot infer split from sample filename: {name}")


def load_records(samples_dir):
    manifest_path = samples_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_rows = read_jsonl(manifest_path)
        records = []
        for row in manifest_rows:
            output_path = Path(row["output_path"])
            sample_rel = output_path.name
            candidate = samples_dir / sample_rel
            if not candidate.exists():
                try:
                    sample_rel = output_path.resolve().relative_to(samples_dir.resolve())
                except ValueError:
                    sample_rel = output_path.name
            records.append(
                {
                    "sample_rel": Path(sample_rel),
                    "source_path": Path(row["source_path"]),
                    "split": row.get("split"),
                }
            )
        return records

    records = []
    for wav_path in sorted(samples_dir.rglob("*.wav")):
        inferred = infer_from_sample_name(wav_path)
        records.append(
            {
                "sample_rel": wav_path.relative_to(samples_dir),
                "source_path": inferred["source_path"],
                "split": inferred["split"],
            }
        )
    return records


def infer_split(records, requested_split):
    if requested_split:
        return requested_split
    splits = {record["split"] for record in records if record.get("split")}
    if len(splits) == 1:
        return splits.pop()
    if not splits:
        raise ValueError("Could not infer split. Pass --split.")
    raise ValueError(f"Multiple splits found in samples: {sorted(splits)}. Pass --split.")


def yaml_load(path):
    if yaml is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def infer_root_path(samples_dir, records):
    for record in records:
        source_path = record["source_path"]
        if source_path.is_absolute() and source_path.exists():
            return None

    config_candidates = []
    config_candidates.extend(samples_dir.parent.glob("*.yml"))
    config_candidates.extend(samples_dir.parent.glob("*.yaml"))
    config_candidates.extend((REPO_ROOT / "Configs").glob("*.yml"))
    config_candidates.extend((REPO_ROOT / "Configs").glob("*.yaml"))

    for config_path in config_candidates:
        try:
            config = yaml_load(config_path)
        except Exception:
            continue
        if not isinstance(config, dict):
            continue

        roots = []
        if "root_path" in config:
            roots.append(config["root_path"])
        data_params = config.get("data_params", {})
        if isinstance(data_params, dict) and "root_path" in data_params:
            roots.append(data_params["root_path"])

        for root in roots:
            if root is None:
                continue
            root_path = Path(root)
            if all((root_path / record["source_path"]).exists() for record in records[:10]):
                return root_path

    raise FileNotFoundError(
        "Could not infer dataset root for source wavs. Pass --root-path."
    )


def resolve_source_path(source_path, root_path):
    if source_path.is_absolute():
        return source_path
    if root_path is None:
        return source_path
    return root_path / source_path


def mirror_ground_truth(records, samples_dir, ground_truth_dir, root_path, copy, refresh):
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    reused = 0
    missing = []

    for record in records:
        sample_path = samples_dir / record["sample_rel"]
        if not sample_path.exists():
            missing.append(str(sample_path))
            continue

        source_path = resolve_source_path(record["source_path"], root_path)
        if not source_path.exists():
            missing.append(str(source_path))
            continue

        dest_path = ground_truth_dir / record["sample_rel"]
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists() or dest_path.is_symlink():
            if not refresh:
                reused += 1
                continue
            if dest_path.is_dir():
                shutil.rmtree(dest_path)
            else:
                dest_path.unlink()

        if copy:
            shutil.copy2(source_path, dest_path)
        else:
            os.symlink(source_path, dest_path)
        linked += 1

    if missing:
        shown = "\n".join(missing[:20])
        more = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise FileNotFoundError(f"Missing sample/source wavs:\n{shown}{more}")

    return linked, reused


def make_flat_eval_dirs(records, samples_dir, ground_truth_dir, temp_dir):
    gen_dir = temp_dir / "generated"
    gt_dir = temp_dir / "ground_truth"
    gen_dir.mkdir()
    gt_dir.mkdir()

    used_names = set()
    for record in records:
        rel = record["sample_rel"]
        flat_name = rel.name if rel.name not in used_names else "__".join(rel.parts)
        if flat_name in used_names:
            raise ValueError(f"Duplicate flattened sample name: {flat_name}")
        used_names.add(flat_name)
        os.symlink((samples_dir / rel).resolve(), gen_dir / flat_name)
        os.symlink((ground_truth_dir / rel).resolve(), gt_dir / flat_name)

    return gen_dir, gt_dir


def run_command(command):
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.returncode, proc.stdout


def summarize_log_f0_output(output):
    for line in reversed(output.splitlines()):
        if line.startswith("Average:"):
            return line + "\n"
    return output


def collect_metric_artifacts(eval_output_dir, samples_dir):
    artifact_map = {
        "utt2mcd.log": "eval_tts_utt2mcd.log",
        "evaluation_results.txt": "eval_tts_mcd_summary.txt",
        "utt2log_f0_rmse": "eval_tts_utt2log_f0_rmse",
        "log_f0_rmse_avg_result.txt": "eval_tts_log_f0_rmse_summary.txt",
    }
    copied = []
    for source_name, dest_name in artifact_map.items():
        source_path = eval_output_dir / source_name
        if not source_path.exists():
            continue
        dest_path = samples_dir / dest_name
        shutil.copy2(source_path, dest_path)
        copied.append(dest_path)
    return copied


def write_results(samples_dir, split, ground_truth_dir, root_path, commands, outputs, artifacts):
    result_path = samples_dir / "eval_tts_results.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("TTS evaluation results\n")
        f.write(f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"samples_dir: {samples_dir}\n")
        f.write(f"split: {split}\n")
        f.write(f"ground_truth_dir: {ground_truth_dir}\n")
        if root_path is not None:
            f.write(f"root_path: {root_path}\n")
        if artifacts:
            f.write("artifacts:\n")
            for artifact in artifacts:
                f.write(f"  {artifact}\n")
        f.write("\n")
        for name, command in commands.items():
            returncode, output = outputs[name]
            f.write(f"[{name}]\n")
            f.write(f"returncode: {returncode}\n")
            f.write("command: " + " ".join(str(part) for part in command) + "\n")
            f.write(output.rstrip() + "\n\n")
    return result_path


def main():
    args = parse_args()
    samples_dir = args.samples_path.resolve()
    if not samples_dir.is_dir():
        raise NotADirectoryError(f"Samples path is not a directory: {samples_dir}")

    records = load_records(samples_dir)
    if not records:
        raise FileNotFoundError(f"No wav samples found in: {samples_dir}")

    split = infer_split(records, args.split)
    root_path = args.root_path.resolve() if args.root_path else infer_root_path(samples_dir, records)
    ground_truth_dir = (
        args.ground_truth_dir.resolve()
        if args.ground_truth_dir
        else REPO_ROOT / "Models" / f"ground_truth_samples_{split}"
    )

    linked, reused = mirror_ground_truth(
        records,
        samples_dir,
        ground_truth_dir,
        root_path,
        args.copy,
        args.refresh_ground_truth,
    )

    commands = {}
    outputs = {}
    artifacts = []
    with tempfile.TemporaryDirectory(prefix="eval_tts_") as temp:
        gen_eval_dir, gt_eval_dir = make_flat_eval_dirs(
            records, samples_dir, ground_truth_dir, Path(temp)
        )

        if not args.skip_mcd:
            commands["mcd"] = [
                sys.executable,
                str(EVAL_DIR / "evaluate_mcd.py"),
                str(gt_eval_dir),
                str(gen_eval_dir),
            ]
            outputs["mcd"] = run_command(commands["mcd"])

        if not args.skip_f0:
            commands["f0"] = [
                sys.executable,
                str(EVAL_DIR / "evaluate_f0.py"),
                str(gen_eval_dir),
                str(gt_eval_dir),
                "--outdir",
                str(gen_eval_dir),
                "--nj",
                str(args.nj),
            ]
            outputs["f0"] = run_command(commands["f0"])
            if outputs["f0"][0] == 0:
                outputs["f0"] = (outputs["f0"][0], summarize_log_f0_output(outputs["f0"][1]))

        artifacts = collect_metric_artifacts(gen_eval_dir, samples_dir)

    result_path = write_results(
        samples_dir, split, ground_truth_dir, root_path, commands, outputs, artifacts
    )

    failed = [name for name, (returncode, _) in outputs.items() if returncode != 0]
    print(f"Mirrored ground truth: {linked} created, {reused} reused")
    print(f"Wrote evaluation results: {result_path}")
    if failed:
        raise RuntimeError(f"Evaluation failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
