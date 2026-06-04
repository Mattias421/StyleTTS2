import argparse
import json
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
from Utils.PLBERT.util import load_plbert
from models import build_model, load_ASR_models, load_F0_models
from text_utils import TextCleaner
from utils import recursive_munch


MEAN, STD = -4, 4
MIN_STYLE_MEL_FRAMES = 80


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ESD speech samples from a StyleTTS2 checkpoint."
    )
    parser.add_argument(
        "-p",
        "--config_path",
        default="Configs/tts_esd.yml",
        help="Path to inference YAML config.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(config):
    configured = str(config.get("device", "auto")).lower()
    if configured in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if configured.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(configured)


def length_to_mask(lengths):
    mask = (
        torch.arange(lengths.max(), device=lengths.device)
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .type_as(lengths)
    )
    return torch.gt(mask + 1, lengths.unsqueeze(1))


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_esd_list(path):
    entries = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_number} must have wav|text|speaker_id")
            entries.append(
                {
                    "relative_wav": parts[0],
                    "phonemized_text": parts[1],
                    "speaker_id": str(parts[2]),
                    "line_number": line_number,
                }
            )
    return entries


def esd_emotion(entry):
    parts = Path(entry["relative_wav"]).parts
    if len(parts) < 4 or parts[0] != "ESD":
        raise ValueError(f"Expected ESD/<speaker>/<emotion>/<wav>, got: {entry['relative_wav']}")
    return parts[2]


def safe_stem(relative_wav):
    stem = Path(relative_wav).with_suffix("").as_posix()
    return stem.replace("/", "_")


def output_name(eval_entry, ref_entry, split_name):
    speaker = int(eval_entry["speaker_id"])
    return (
        f"speaker{speaker:04d}"
        f"__{split_name}_{safe_stem(eval_entry['relative_wav'])}"
        f"__ref_{safe_stem(ref_entry['relative_wav'])}.wav"
    )


def build_to_mel(model_config):
    preprocess_params = model_config.get("preprocess_params", {})
    spect_params = preprocess_params.get("spect_params", {})
    return torchaudio.transforms.MelSpectrogram(
        n_mels=model_config["model_params"].get("n_mels", 80),
        n_fft=spect_params.get("n_fft", 2048),
        win_length=spect_params.get("win_length", 1200),
        hop_length=spect_params.get("hop_length", 300),
    )


def preprocess(wave, to_mel):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    return (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - MEAN) / STD


def approximate_mel_frames(sample_count, to_mel):
    if sample_count <= 0:
        return 0
    return sample_count // to_mel.hop_length + 1


def pad_to_min_mel_frames(audio, to_mel, min_frames):
    frame_count = approximate_mel_frames(len(audio), to_mel)
    if frame_count >= min_frames:
        return audio, frame_count

    min_samples = (min_frames - 1) * to_mel.hop_length
    return np.pad(audio, (0, max(0, min_samples - len(audio)))), frame_count


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_inference_checkpoint(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu")
    params = state["net"]
    for key in model:
        if key not in params:
            print(f"{key} missing from checkpoint")
            continue
        module_state = strip_module_prefix(params[key])
        result = model[key].load_state_dict(module_state, strict=False)
        missing = len(result.missing_keys)
        unexpected = len(result.unexpected_keys)
        note = ""
        if missing or unexpected:
            note = f" ({missing} missing, {unexpected} unexpected)"
        print(f"{key} loaded{note}")
    return state


def load_tts_model(model_config, checkpoint_path, device):
    text_aligner = load_ASR_models(
        model_config.get("ASR_path", False),
        model_config.get("ASR_config", False),
    )
    pitch_extractor = load_F0_models(model_config.get("F0_path", False))
    plbert = load_plbert(model_config.get("PLBERT_dir", False))

    model_params = recursive_munch(model_config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    load_inference_checkpoint(model, checkpoint_path)

    for key in model:
        model[key].eval()
        model[key].to(device)
    return model, model_params


def compute_style(path, model, to_mel, device, sample_rate):
    wave, sr = librosa.load(path, sr=sample_rate)
    trimmed_audio, _ = librosa.effects.trim(wave, top_db=30)
    if sr != sample_rate:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=sample_rate)
        trimmed_audio = librosa.resample(trimmed_audio, orig_sr=sr, target_sr=sample_rate)

    trimmed_frames = approximate_mel_frames(len(trimmed_audio), to_mel)
    if trimmed_frames >= MIN_STYLE_MEL_FRAMES:
        audio = trimmed_audio
    else:
        audio, untrimmed_frames = pad_to_min_mel_frames(wave, to_mel, MIN_STYLE_MEL_FRAMES)
        if untrimmed_frames < MIN_STYLE_MEL_FRAMES:
            print(
                f"Warning: reference {path} trimmed to {trimmed_frames} mel frames; "
                f"padding untrimmed audio to {MIN_STYLE_MEL_FRAMES} frames."
            )
        else:
            print(
                f"Warning: reference {path} trimmed to {trimmed_frames} mel frames; "
                "using untrimmed audio."
            )

    mel_tensor = preprocess(audio, to_mel).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)


def tokens_from_phonemized(text, text_cleaner, device):
    tokens = text_cleaner(text.strip())
    tokens.insert(0, 0)
    return torch.LongTensor(tokens).to(device).unsqueeze(0)


def inference(
    phonemized_text,
    ref_s,
    model,
    model_params,
    sampler,
    text_cleaner,
    device,
    alpha,
    beta,
    diffusion_steps,
    embedding_scale,
):
    tokens = tokens_from_phonemized(phonemized_text, text_cleaner, device)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(
            noise=torch.randn((1, 256), device=device).unsqueeze(1),
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            features=ref_s,
            num_steps=diffusion_steps,
        ).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data), device=device)
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            frame_count = int(pred_dur[i].data)
            pred_aln_trg[i, c_frame : c_frame + frame_count] = 1
            c_frame += frame_count

        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0)
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        f0_pred, n_pred = model.predictor.F0Ntrain(en, s)

        asr = t_en @ pred_aln_trg.unsqueeze(0)
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, f0_pred, n_pred, ref.squeeze().unsqueeze(0))

    wav = out.squeeze().cpu().numpy()
    if wav.shape[-1] > 50:
        wav = wav[..., :-50]
    return wav


def resolve_split_paths(config):
    split_name = str(config.get("split", "val"))
    if split_name not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got: {split_name}")

    data_dir = Path(config.get("data_dir", "Data"))
    text_list = Path(config.get("text_list", data_dir / f"{split_name}_list_esd.txt"))
    ref_list = Path(config.get("ref_list", data_dir / f"ref_{split_name}_list_esd.txt"))
    return split_name, text_list, ref_list


def resolve_output_dir(config, model_config):
    if "output_dir" in config and config["output_dir"]:
        return Path(config["output_dir"])

    log_dir = model_config.get("log_dir")
    if log_dir:
        model_dir = Path(log_dir)
    else:
        model_dir = Path(config["checkpoint"]).parent
    return model_dir / "samples"


def validate_paired_entries(eval_entries, ref_entries, text_list, ref_list):
    if len(eval_entries) != len(ref_entries):
        raise ValueError(
            f"{text_list} has {len(eval_entries)} rows but "
            f"{ref_list} has {len(ref_entries)} rows"
        )

    for index, (eval_entry, ref_entry) in enumerate(zip(eval_entries, ref_entries), start=1):
        if eval_entry["speaker_id"] != ref_entry["speaker_id"]:
            raise ValueError(
                f"Speaker mismatch at row {index}: "
                f"{eval_entry['speaker_id']} != {ref_entry['speaker_id']}"
            )
        eval_emotion = esd_emotion(eval_entry)
        ref_emotion = esd_emotion(ref_entry)
        if eval_emotion != ref_emotion:
            raise ValueError(
                f"Emotion mismatch at row {index}: {eval_emotion} != {ref_emotion}"
            )
        if eval_entry["relative_wav"] == ref_entry["relative_wav"]:
            raise ValueError(f"Reference repeats evaluation wav at row {index}")


def main():
    args = parse_args()
    config = load_yaml(args.config_path)
    set_seed(int(config.get("seed", 0)))
    device = resolve_device(config)

    model_config_path = config["model_config"]
    model_config = load_yaml(model_config_path)
    sample_rate = int(config.get("sample_rate", model_config["preprocess_params"].get("sr", 24000)))
    root_path = Path(config["root_path"])
    split_name, text_list, ref_list = resolve_split_paths(config)
    eval_entries = parse_esd_list(text_list)
    ref_entries = parse_esd_list(ref_list)
    validate_paired_entries(eval_entries, ref_entries, text_list, ref_list)
    paired_entries = list(zip(eval_entries, ref_entries))
    limit = config.get("limit")
    if limit is not None:
        paired_entries = paired_entries[: int(limit)]

    output_dir = resolve_output_dir(config, model_config)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model config: {model_config_path}")
    print(f"Loading checkpoint: {config['checkpoint']}")
    print(f"Using device: {device}")
    print(f"Using {split_name} split: {text_list}")
    print(f"Using references: {ref_list}")
    print(f"Writing samples to: {output_dir}")
    model, model_params = load_tts_model(model_config, config["checkpoint"], device)
    to_mel = build_to_mel(model_config)
    text_cleaner = TextCleaner()
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )

    alpha = float(config.get("alpha", 0.3))
    beta = float(config.get("beta", 0.7))
    diffusion_steps = int(config.get("diffusion_steps", 5))
    embedding_scale = float(config.get("embedding_scale", 1.0))

    style_cache = {}
    manifest_path = output_dir / "manifest.jsonl"

    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for index, (eval_entry, ref_entry) in enumerate(paired_entries, start=1):
            ref_path = root_path / ref_entry["relative_wav"]
            if ref_entry["relative_wav"] not in style_cache:
                style_cache[ref_entry["relative_wav"]] = compute_style(
                    ref_path, model, to_mel, device, sample_rate
                )
            ref_s = style_cache[ref_entry["relative_wav"]]

            wav = inference(
                eval_entry["phonemized_text"],
                ref_s,
                model,
                model_params,
                sampler,
                text_cleaner,
                device,
                alpha,
                beta,
                diffusion_steps,
                embedding_scale,
            )

            output_path = output_dir / output_name(eval_entry, ref_entry, split_name)
            sf.write(output_path, wav, sample_rate)
            row = {
                "index": index,
                "split": split_name,
                "text_list": str(text_list),
                "ref_list": str(ref_list),
                "source_path": eval_entry["relative_wav"],
                "ref_path": ref_entry["relative_wav"],
                "speaker_id": eval_entry["speaker_id"],
                "output_path": str(output_path),
                "alpha": alpha,
                "beta": beta,
                "diffusion_steps": diffusion_steps,
                "embedding_scale": embedding_scale,
                "sample_rate": sample_rate,
            }
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{index}/{len(paired_entries)}] {eval_entry['relative_wav']} "
                f"-> {output_path}"
            )

    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
