#!/usr/bin/env python3
import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import phonemizer


ENGLISH_SPEAKERS = [f"{speaker:04d}" for speaker in range(11, 21)]
EMOTIONS = ["Neutral", "Angry", "Happy", "Sad", "Surprise"]
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


def validate_coverage(utterances: dict[tuple[int, str, str], Utterance]) -> list[int]:
    sentence_ids = sorted({key[0] for key in utterances})
    for sentence_id in sentence_ids:
        for speaker in ENGLISH_SPEAKERS:
            for emotion in EMOTIONS:
                key = (sentence_id, speaker, emotion)
                if key not in utterances:
                    raise ValueError(f"Missing utterance for sentence/speaker/emotion: {key}")
    return sentence_ids


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


def write_manifest(path: Path, records: list[Utterance], ipa_phonemizer: IpaPhonemizer) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            ipa = ipa_phonemizer.phonemize(record.text)
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

    utterances = read_utterances(esd_root, transcript_root, args.path_prefix)
    sentence_ids = validate_coverage(utterances)
    splits = split_sentence_ids(sentence_ids, args.seed)
    assert_disjoint(splits)

    train_records = collect_records(splits["train"], utterances)
    val_records = collect_records(splits["val"], utterances)
    test_records = collect_records(splits["test"], utterances)
    ref_val_records = collect_reference_records(val_records, splits["ref_val"], utterances)
    ref_test_records = collect_reference_records(test_records, splits["ref_test"], utterances)

    assert_reference_alignment(val_records, ref_val_records)
    assert_reference_alignment(test_records, ref_test_records)

    ipa_phonemizer = IpaPhonemizer()
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


if __name__ == "__main__":
    main()
