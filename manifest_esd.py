#!/usr/bin/env python3
import argparse
import contextlib
import json
import random
import re
import string
import wave
from dataclasses import dataclass
from pathlib import Path

import phonemizer


ENGLISH_SPEAKERS = [f"{speaker:04d}" for speaker in range(11, 21)]
EMOTIONS = ["Neutral", "Angry", "Happy", "Sad", "Surprise"]
EVAL_SPLITS = ["val", "test", "ref_val", "ref_test"]
TRAIN_SENTENCES = 180
VAL_SENTENCES = 40
TEST_SENTENCES = 40
REF_VAL_SENTENCES = 40
REF_TEST_SENTENCES = 40


@dataclass(frozen=True)
class Utterance:
    wav_path: str
    source_wav_path: Path
    text: str
    speaker: str
    speaker_id: int
    emotion: str
    base_sentence_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ESD English IPA manifests.")
    parser.add_argument("--esd-root", default="ESD", help="Directory containing ESD wav files.")
    parser.add_argument(
        "--transcript-root",
        default="ESD_og",
        help="Directory containing ESD transcript files.",
    )
    parser.add_argument("--out-dir", default=".", help="Directory for output manifest files.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for split selection.")
    parser.add_argument(
        "--path-prefix",
        default="ESD",
        help="Path prefix written into manifest wav paths.",
    )
    parser.add_argument(
        "--mode",
        choices=["legacy", "minutes"],
        default="legacy",
        help="Legacy reproduces the current 180/40/40/40/40 split. Minutes creates the new budgeted splits.",
    )
    parser.add_argument(
        "--target-minutes",
        default="1,5,15,30,60",
        help="Comma-separated per-speaker training budgets used in minutes mode.",
    )
    parser.add_argument(
        "--eval-sentence-ids-per-split",
        type=int,
        default=4,
        help="Number of sentence IDs per validation/test split in minutes mode.",
    )
    parser.add_argument(
        "--budget-out-prefix",
        default="esd",
        help="Prefix used when writing minute-budget manifest filenames.",
    )
    parser.add_argument(
        "--metadata-name",
        default="esd_minutes_split_plan.json",
        help="Metadata file written in minutes mode.",
    )
    parser.add_argument(
        "--source-manifest-dir",
        default="",
        help="Optional directory containing existing ESD manifests to reuse as phonemized source text in minutes mode.",
    )
    return parser.parse_args()


def base_sentence_id(utt_id: str) -> int:
    utt_number = int(utt_id.rsplit("_", 1)[1])
    return ((utt_number - 1) % 350) + 1


def read_utterances(esd_root: Path, transcript_root: Path, path_prefix: str) -> dict[tuple[int, str, str], Utterance]:
    utterances: dict[tuple[int, str, str], Utterance] = {}

    for speaker_id, speaker in enumerate(ENGLISH_SPEAKERS):
        transcript_path = transcript_root / speaker / f"{speaker}.txt"
        if not transcript_path.exists():
            raise FileNotFoundError(f"Missing transcript: {transcript_path}")

        with transcript_path.open("r", encoding="utf-8") as transcript:
            for line_number, line in enumerate(transcript, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) != 3:
                    raise ValueError(
                        f"Expected 3 tab-separated fields in {transcript_path}:{line_number}"
                    )

                utt_id, text, emotion = parts
                emotion = emotion.strip()
                if emotion not in EMOTIONS:
                    continue

                sentence_id = base_sentence_id(utt_id)
                source_wav_path = esd_root / speaker / emotion / f"{utt_id}.wav"
                if not source_wav_path.exists():
                    raise FileNotFoundError(f"Missing wav: {source_wav_path}")

                key = (sentence_id, speaker, emotion)
                if key in utterances:
                    raise ValueError(f"Duplicate utterance for sentence/speaker/emotion: {key}")

                wav_path = f"{path_prefix.rstrip('/')}/{speaker}/{emotion}/{utt_id}.wav"
                utterances[key] = Utterance(
                    wav_path=wav_path,
                    source_wav_path=source_wav_path,
                    text=text.strip(),
                    speaker=speaker,
                    speaker_id=speaker_id,
                    emotion=emotion,
                    base_sentence_id=sentence_id,
                )

    return utterances


def read_utterances_from_manifests(
    esd_root: Path,
    manifest_root: Path,
) -> dict[tuple[int, str, str], Utterance]:
    utterances: dict[tuple[int, str, str], Utterance] = {}

    manifest_names = [
        "train_list_esd.txt",
        "val_list_esd.txt",
        "test_list_esd.txt",
        "ref_val_list_esd.txt",
        "ref_test_list_esd.txt",
    ]

    for manifest_name in manifest_names:
        manifest_path = manifest_root / manifest_name
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing source manifest: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as manifest:
            for line_number, line in enumerate(manifest, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue

                parts = line.split("|")
                if len(parts) != 3:
                    raise ValueError(
                        f"Expected 3 pipe-separated fields in {manifest_path}:{line_number}"
                    )

                wav_path, text, speaker_id_text = parts
                speaker_id_text = speaker_id_text.strip()
                source_wav_path = esd_root / wav_path
                if not source_wav_path.exists():
                    raise FileNotFoundError(f"Missing wav: {source_wav_path}")

                utt_id = Path(wav_path).stem
                sentence_id = base_sentence_id(utt_id)
                wav_parts = Path(wav_path).parts
                if len(wav_parts) < 4:
                    raise ValueError(f"Unexpected wav path in source manifest: {wav_path}")
                speaker = wav_parts[1]
                emotion = wav_parts[2]
                key = (sentence_id, speaker, emotion)
                if key in utterances:
                    continue

                utterances[key] = Utterance(
                    wav_path=wav_path,
                    source_wav_path=source_wav_path,
                    text=text.strip(),
                    speaker=speaker,
                    speaker_id=int(speaker_id_text),
                    emotion=emotion,
                    base_sentence_id=sentence_id,
                )

    return utterances


def validate_coverage(utterances: dict[tuple[int, str, str], Utterance]) -> list[int]:
    sentence_ids = sorted({key[0] for key in utterances})
    for sentence_id in sentence_ids:
        for speaker in ENGLISH_SPEAKERS:
            for emotion in EMOTIONS:
                key = (sentence_id, speaker, emotion)
                if key not in utterances:
                    raise ValueError(f"Missing utterance for sentence/speaker/emotion: {key}")
    return sentence_ids


def read_duration_minutes(wav_path: Path) -> float:
    with contextlib.closing(wave.open(str(wav_path), "rb")) as wav_file:
        return wav_file.getnframes() / float(wav_file.getframerate()) / 60.0


def build_sentence_minutes(
    utterances: dict[tuple[int, str, str], Utterance]
) -> dict[int, float]:
    sentence_minutes: dict[int, float] = {}
    for utterance in utterances.values():
        sentence_minutes.setdefault(utterance.base_sentence_id, 0.0)
        sentence_minutes[utterance.base_sentence_id] += read_duration_minutes(
            utterance.source_wav_path
        )

    speaker_count = len(ENGLISH_SPEAKERS)
    return {
        sentence_id: total_minutes / speaker_count
        for sentence_id, total_minutes in sentence_minutes.items()
    }


def split_sentence_ids(sentence_ids: list[int], seed: int) -> dict[str, list[int]]:
    required = TRAIN_SENTENCES + VAL_SENTENCES + TEST_SENTENCES + REF_VAL_SENTENCES + REF_TEST_SENTENCES
    if len(sentence_ids) < required:
        raise ValueError(f"Need {required} sentence IDs, found {len(sentence_ids)}")

    shuffled = list(sentence_ids)
    random.Random(seed).shuffle(shuffled)

    cursor = 0
    split_sizes = {
        "train": TRAIN_SENTENCES,
        "val": VAL_SENTENCES,
        "test": TEST_SENTENCES,
        "ref_val": REF_VAL_SENTENCES,
        "ref_test": REF_TEST_SENTENCES,
    }
    splits: dict[str, list[int]] = {}
    for split_name, split_size in split_sizes.items():
        splits[split_name] = shuffled[cursor : cursor + split_size]
        cursor += split_size
    return splits


def split_eval_sentence_ids(
    sentence_ids: list[int],
    sentence_minutes: dict[int, float],
    eval_sentence_ids_per_split: int,
    seed: int,
) -> dict[str, list[int]]:
    required = eval_sentence_ids_per_split * len(EVAL_SPLITS)
    if len(sentence_ids) < required:
        raise ValueError(f"Need at least {required} sentence IDs, found {len(sentence_ids)}")

    ordered = sorted(sentence_ids, key=lambda sentence_id: (sentence_minutes[sentence_id], sentence_id))
    bin_size = (len(ordered) + eval_sentence_ids_per_split - 1) // eval_sentence_ids_per_split
    bins = [
        ordered[index : index + bin_size]
        for index in range(0, len(ordered), bin_size)
    ]
    if len(bins) < eval_sentence_ids_per_split:
        raise ValueError("Could not build enough duration bins for evaluation selection")

    splits: dict[str, list[int]] = {split_name: [] for split_name in EVAL_SPLITS}
    for bin_index, bin_sentence_ids in enumerate(bins[:eval_sentence_ids_per_split]):
        shuffled_bin = list(bin_sentence_ids)
        random.Random(seed + 1000 + bin_index).shuffle(shuffled_bin)
        selected = shuffled_bin[: len(EVAL_SPLITS)]
        if len(selected) < len(EVAL_SPLITS):
            raise ValueError("Evaluation bin is too small")
        for split_name, sentence_id in zip(EVAL_SPLITS, selected):
            splits[split_name].append(sentence_id)

    for split_name in splits:
        splits[split_name].sort()
    return splits


def build_budget_prefixes(
    sentence_ids: list[int],
    sentence_minutes: dict[int, float],
    target_minutes: list[float],
    seed: int,
) -> dict[str, dict[str, object]]:
    ordered = list(sentence_ids)
    random.Random(seed).shuffle(ordered)

    prefixes: dict[str, dict[str, object]] = {}
    cumulative_minutes = 0.0
    target_index = 0
    sorted_targets = sorted(target_minutes)

    for index, sentence_id in enumerate(ordered, start=1):
        cumulative_minutes += sentence_minutes[sentence_id]
        while target_index < len(sorted_targets) and cumulative_minutes >= sorted_targets[target_index]:
            target = sorted_targets[target_index]
            label = f"{int(target)}m"
            prefixes[label] = {
                "target_minutes": target,
                "selected_sentence_ids": ordered[:index],
                "actual_minutes_per_speaker": cumulative_minutes,
            }
            target_index += 1

    if target_index != len(sorted_targets):
        missing = sorted_targets[target_index:]
        raise ValueError(f"Unable to reach target minute budgets: {missing}")

    return prefixes


def write_metadata(path: Path, metadata: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2, sort_keys=True)
        output.write("\n")


def collect_records(
    sentence_ids: list[int],
    utterances: dict[tuple[int, str, str], Utterance],
) -> list[Utterance]:
    records: list[Utterance] = []
    for sentence_id in sentence_ids:
        for speaker in ENGLISH_SPEAKERS:
            for emotion in EMOTIONS:
                records.append(utterances[(sentence_id, speaker, emotion)])
    return records


def collect_reference_records(
    eval_records: list[Utterance],
    reference_sentence_ids: list[int],
    utterances: dict[tuple[int, str, str], Utterance],
) -> list[Utterance]:
    expected_eval_sentences = len(reference_sentence_ids)
    eval_rows_per_sentence = len(ENGLISH_SPEAKERS) * len(EMOTIONS)
    if len(eval_records) != expected_eval_sentences * eval_rows_per_sentence:
        raise ValueError("Reference and evaluation split sizes do not match")

    reference_records: list[Utterance] = []
    for index, eval_record in enumerate(eval_records):
        reference_sentence_id = reference_sentence_ids[index // eval_rows_per_sentence]
        reference_records.append(
            utterances[(reference_sentence_id, eval_record.speaker, eval_record.emotion)]
        )
    return reference_records


class IpaPhonemizer:
    def __init__(self) -> None:
        self.backend = phonemizer.backend.EspeakBackend(
            language="en-us",
            preserve_punctuation=True,
            with_stress=True,
        )
        self.cache: dict[str, str] = {}

    def phonemize(self, text: str) -> str:
        text = text.strip()
        ipa = self.cache.get(text)
        if ipa is None:
            ipa = self.backend.phonemize([text])[0].strip()
            self.cache[text] = ipa
        return ipa


def normalize_styletts2_phoneme_text(text: str) -> str:
    """Match the token spacing StyleTTS2 expects in its filelists."""
    spaced = re.sub(rf"\s*([{re.escape(string.punctuation)}])\s*", r" \1 ", text.strip())
    return re.sub(r"\s+", " ", spaced).strip()


def write_manifest(
    path: Path,
    records: list[Utterance],
    ipa_phonemizer: IpaPhonemizer | None,
) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            raw_text = ipa_phonemizer.phonemize(record.text) if ipa_phonemizer is not None else record.text
            ipa = normalize_styletts2_phoneme_text(raw_text)
            output.write(f"{record.wav_path}|{ipa}|{record.speaker_id}\n")


def assert_disjoint(splits: dict[str, list[int]]) -> None:
    seen: set[int] = set()
    for split_name, sentence_ids in splits.items():
        duplicate_ids = seen.intersection(sentence_ids)
        if duplicate_ids:
            raise ValueError(f"{split_name} overlaps with earlier splits: {sorted(duplicate_ids)}")
        seen.update(sentence_ids)


def assert_reference_alignment(eval_records: list[Utterance], reference_records: list[Utterance]) -> None:
    if len(eval_records) != len(reference_records):
        raise ValueError("Evaluation and reference record counts differ")

    for index, (eval_record, reference_record) in enumerate(zip(eval_records, reference_records)):
        if eval_record.speaker_id != reference_record.speaker_id:
            raise ValueError(f"Speaker mismatch at row {index}")
        if eval_record.emotion != reference_record.emotion:
            raise ValueError(f"Emotion mismatch at row {index}")
        if eval_record.base_sentence_id == reference_record.base_sentence_id:
            raise ValueError(f"Reference sentence repeats evaluation sentence at row {index}")


def main() -> None:
    args = parse_args()
    esd_root = Path(args.esd_root)
    transcript_root = Path(args.transcript_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_manifest_root = Path(args.source_manifest_dir) if args.source_manifest_dir else None
    if source_manifest_root is not None:
        utterances = read_utterances_from_manifests(esd_root, source_manifest_root)
    else:
        utterances = read_utterances(esd_root, transcript_root, args.path_prefix)
    sentence_ids = validate_coverage(utterances)
    ipa_phonemizer = None if source_manifest_root is not None else IpaPhonemizer()
    if args.mode == "legacy":
        splits = split_sentence_ids(sentence_ids, args.seed)
        assert_disjoint(splits)

        train_records = collect_records(splits["train"], utterances)
        val_records = collect_records(splits["val"], utterances)
        test_records = collect_records(splits["test"], utterances)
        ref_val_records = collect_reference_records(val_records, splits["ref_val"], utterances)
        ref_test_records = collect_reference_records(test_records, splits["ref_test"], utterances)

        assert_reference_alignment(val_records, ref_val_records)
        assert_reference_alignment(test_records, ref_test_records)

        manifests = {
            "train_list_esd.txt": train_records,
            "val_list_esd.txt": val_records,
            "test_list_esd.txt": test_records,
            "ref_val_list_esd.txt": ref_val_records,
            "ref_test_list_esd.txt": ref_test_records,
        }
        for filename, records in manifests.items():
            write_manifest(out_dir / filename, records, ipa_phonemizer)

        for split_name in ["train", "val", "test", "ref_val", "ref_test"]:
            print(f"{split_name}: {len(splits[split_name])} sentence IDs {splits[split_name]}")
        for filename, records in manifests.items():
            print(f"{filename}: {len(records)} rows")
        return

    sentence_minutes = build_sentence_minutes(utterances)
    target_minutes = [float(value) for value in args.target_minutes.split(",") if value.strip()]
    eval_splits = split_eval_sentence_ids(
        sentence_ids,
        sentence_minutes,
        args.eval_sentence_ids_per_split,
        args.seed,
    )
    assert_disjoint(eval_splits)

    eval_records: dict[str, list[Utterance]] = {}
    for split_name in EVAL_SPLITS:
        eval_records[split_name] = collect_records(eval_splits[split_name], utterances)

    ref_records = {
        "ref_val": collect_reference_records(eval_records["val"], eval_splits["ref_val"], utterances),
        "ref_test": collect_reference_records(eval_records["test"], eval_splits["ref_test"], utterances),
    }

    assert_reference_alignment(eval_records["val"], ref_records["ref_val"])
    assert_reference_alignment(eval_records["test"], ref_records["ref_test"])

    remaining_sentence_ids = [sentence_id for sentence_id in sentence_ids if sentence_id not in {sid for split_ids in eval_splits.values() for sid in split_ids}]
    budget_prefixes = build_budget_prefixes(remaining_sentence_ids, sentence_minutes, target_minutes, args.seed)

    metadata = {
        "mode": "minutes",
        "seed": args.seed,
        "eval_sentence_ids_per_split": args.eval_sentence_ids_per_split,
        "sentence_id_minutes": {str(sentence_id): sentence_minutes[sentence_id] for sentence_id in sentence_ids},
        "eval_splits": eval_splits,
        "budgets": {},
    }

    for split_name in ["val", "test"]:
        filename = f"{split_name}_list_{args.budget_out_prefix}_eval{args.eval_sentence_ids_per_split}.txt"
        write_manifest(out_dir / filename, eval_records[split_name], ipa_phonemizer)
        print(f"{filename}: {len(eval_records[split_name])} rows")

    ref_eval_filenames = {
        "ref_val": f"ref_val_list_{args.budget_out_prefix}_eval{args.eval_sentence_ids_per_split}.txt",
        "ref_test": f"ref_test_list_{args.budget_out_prefix}_eval{args.eval_sentence_ids_per_split}.txt",
    }
    for split_name, filename in ref_eval_filenames.items():
        write_manifest(out_dir / filename, ref_records[split_name], ipa_phonemizer)
        print(f"{filename}: {len(ref_records[split_name])} rows")

    for label, details in budget_prefixes.items():
        selected_sentence_ids = details["selected_sentence_ids"]
        train_records = collect_records(selected_sentence_ids, utterances)
        train_filename = f"train_list_{args.budget_out_prefix}_{label}.txt"
        write_manifest(out_dir / train_filename, train_records, ipa_phonemizer)
        metadata["budgets"][label] = {
            "target_minutes": details["target_minutes"],
            "actual_minutes_per_speaker": details["actual_minutes_per_speaker"],
            "selected_sentence_ids": selected_sentence_ids,
            "train_rows": len(train_records),
            "train_sentence_ids": len(selected_sentence_ids),
            "train_manifest": train_filename,
        }
        print(
            f"{train_filename}: {len(train_records)} rows "
            f"({len(selected_sentence_ids)} sentence IDs, "
            f"{details['actual_minutes_per_speaker']:.2f} min/speaker)"
        )

    metadata["eval_manifest_prefix"] = args.budget_out_prefix
    metadata["eval_manifest_suffix"] = f"eval{args.eval_sentence_ids_per_split}"
    write_metadata(out_dir / args.metadata_name, metadata)
    print(f"{args.metadata_name}: wrote split metadata")


if __name__ == "__main__":
    main()
