import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SERIES = {
    "base": {
        "directory": "base",
        "base_config": "base.yml",
    },
    "base_slm": {
        "directory": "base_slm",
        "base_config": "base_slm.yml",
    },
    "dt010": {
        "directory": "dt010",
        "base_config": "dt010.yml",
    },
    "dt050_l1": {
        "directory": "dt050_l1",
        "base_config": "dt050_l1.yml",
        "slm_config": "dt050_l1_slm.yml",
        "slm_start": 3,
    },
    "dt050_l2": {
        "directory": "dt050_l2",
        "base_config": "dt050_l2.yml",
        "slm_config": "dt050_l2_slm.yml",
        "slm_start": 3,
    },
    "dt100": {
        "directory": "dt100",
        "base_config": "dt100.yml",
        "slm_config": "dt100_slm.yml",
        "slm_start": 3,
        "model_overrides": {
            "cde": {
                "params": {
                    "shared_vector_field": True,
                }
            }
        },
    },
    "dt100_l4": {
        "directory": "dt100_l4",
        "base_config": "dt100_l4.yml",
        "model_overrides": {
            "cde": {
                "params": {
                    "shared_vector_field": True,
                }
            }
        },
    },
    "dt100_l4_slm": {
        "directory": "dt100_l4_slm",
        "base_config": "dt100_l4_slm.yml",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthesize and evaluate every existing checkpoint in exp4."
    )
    parser.add_argument("--gpu", required=True)
    parser.add_argument("series", nargs="+", choices=SERIES)
    return parser.parse_args()


def checkpoint_epoch(path):
    return int(path.stem.rsplit("_", 1)[1])


def config_for_epoch(series_dir, spec, epoch):
    config_name = spec["base_config"]
    if "slm_config" in spec and epoch >= spec["slm_start"]:
        config_name = spec["slm_config"]
    return series_dir / config_name


def complete(samples_dir):
    manifest = samples_dir / "manifest.jsonl"
    result = samples_dir / "eval_tts_results.txt"
    if not manifest.exists() or not result.exists():
        return False
    wav_count = sum(1 for _ in samples_dir.glob("*.wav"))
    manifest_count = sum(1 for _ in manifest.open(encoding="utf-8"))
    return wav_count == 100 and manifest_count == 100


def run(command, log_path, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def inference_config(model_config, checkpoint, output_dir, split, spec):
    config = {
        "model_config": str(model_config.relative_to(REPO_ROOT)),
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "root_path": "/store/store4/data",
        "split": split,
        "text_list": f"Data/mini_{split}_list_esd.txt",
        "ref_list": f"Data/mini_ref_{split}_list_esd.txt",
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "device": "cuda",
        "sample_rate": 24000,
        "alpha": 0.3,
        "beta": 0.7,
        "diffusion_steps": 5,
        "embedding_scale": 1.0,
        "trim_start_samples": 2400,
        "trim_end_samples": 2400,
        "seed": 0,
    }
    if "model_overrides" in spec:
        config["model_overrides"] = spec["model_overrides"]
    return config


def main():
    args = parse_args()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    for series_name in args.series:
        spec = SERIES[series_name]
        series_dir = REPO_ROOT / "Models" / "exp4_stacked" / spec["directory"]
        checkpoints = sorted(
            series_dir.glob("epoch_2nd_*.pth"),
            key=checkpoint_epoch,
        )
        for checkpoint in checkpoints:
            epoch = checkpoint_epoch(checkpoint)
            epoch_dir = series_dir / "per_epoch" / f"epoch_{epoch:05d}"
            model_config = config_for_epoch(series_dir, spec, epoch)
            metadata = {
                "series": series_name,
                "epoch": epoch,
                "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
                "model_config": str(model_config.relative_to(REPO_ROOT)),
            }
            epoch_dir.mkdir(parents=True, exist_ok=True)
            (epoch_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
            )

            for split in ("val", "test"):
                samples_dir = epoch_dir / f"samples_{split}_mini"
                if complete(samples_dir):
                    continue

                config = inference_config(
                    model_config,
                    checkpoint,
                    samples_dir,
                    split,
                    spec,
                )
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".yml",
                    encoding="utf-8",
                    delete=False,
                ) as config_file:
                    yaml.safe_dump(config, config_file, sort_keys=False)
                    config_path = config_file.name
                try:
                    run(
                        [str(PYTHON), "text_to_speech.py", "-p", config_path],
                        epoch_dir / f"generate_{split}_mini.log",
                        env=env,
                    )
                    run(
                        [
                            str(PYTHON),
                            "eval_tts.py",
                            str(samples_dir.relative_to(REPO_ROOT)),
                            "--root-path",
                            "/store/store4/data",
                            "--split",
                            split,
                            "--nj",
                            "8",
                        ],
                        epoch_dir / f"eval_{split}_mini.log",
                    )
                finally:
                    Path(config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
