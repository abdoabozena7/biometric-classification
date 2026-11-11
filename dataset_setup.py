"""Prepare real data without changing any supplied face image."""

import hashlib
import io
import json
import random
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import soundfile as sf
from PIL import Image, ImageOps

from config import (COMMANDS, FACE_ROOT, IMAGE_EXTENSIONS, LIVE, MANIFESTS,
                    PROCESSED, RAW, ROLES, ROOT, SEED, UNSUPPORTED, make_folders)
from features import decoded_hash, image_signature, load_audio, load_image

VOICE_DATASET = "pollen-robotics/speech-commands-v0.02"
VOICE_REVISION = "278418261b4161f6603e06431fa05f9416aaee99"
FVC_PAGE = "https://bias.csr.unibo.it/fvc2004/download.asp"
FVC_ZIP = "http://bias.csr.unibo.it/fvc2004/Downloads/DB1_B.zip"


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_people(face_root=FACE_ROOT):
    people = sorted((folder for folder in Path(face_root).iterdir()
                     if folder.is_dir() and any(
                         path.suffix.lower() in IMAGE_EXTENSIONS
                         for path in folder.iterdir())),
                    key=lambda folder: folder.name.casefold())
    if len(people) != 3:
        raise ValueError(f"Expected exactly three face folders; found {len(people)}.")
    return {f"user_{number:02d}": folder
            for number, folder in enumerate(people, start=1)}


def inspect_files():
    make_folders()
    records = []
    for user_id, folder in discover_people().items():
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({"user_id": user_id, "display_name": folder.name,
                                "source_path": str(path), "sha256": file_hash(path)})
    counts = Counter(record["user_id"] for record in records)
    duplicates = [value for value, count in
                  Counter(record["sha256"] for record in records).items() if count > 1]
    report = {"counts": dict(counts), "exact_duplicate_hashes": duplicates,
              "decoding_checked": False, "files": records}
    (MANIFESTS / "initial_face_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"counts": dict(counts), "duplicate_groups": len(duplicates)}, indent=2))
    return report


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def local_path(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def get_json(url, **params):
    response = requests.get(url, params=params, timeout=90)
    response.raise_for_status()
    return response.json()


def split_items(items, train_count, validation_count, seed=SEED):
    items = list(items)
    random.Random(seed).shuffle(items)
    return [(item, "train" if index < train_count else
             "validation" if index < train_count + validation_count else "test")
            for index, item in enumerate(items)]


def sample_record(user_id, display_name, modality, source, prepared, split,
                  source_identity, source_identifier, command="", source_split="custom"):
    return {"sample_id": f"{modality}:{user_id}:{source_identifier}",
            "user_id": user_id, "display_name": display_name, "modality": modality,
            "split": split, "path": local_path(prepared), "source_path": str(source),
            "source_identity": str(source_identity), "source_identifier": source_identifier,
            "source_split": source_split, "command": command,
            "sha256": file_hash(prepared), "decoded_sha256": decoded_hash(prepared, modality)}


def prepare_faces(people):
    records, warnings, rejected, signatures = [], [], [], {}
    seen_files, seen_pixels = set(), set()
    for user_id, folder in people.items():
        valid = []
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                load_image(path)
                raw_hash, pixels_hash = file_hash(path), decoded_hash(path, "face")
                if raw_hash in seen_files or pixels_hash in seen_pixels:
                    rejected.append({"path": local_path(path), "reason": "exact duplicate"})
                    continue
                seen_files.add(raw_hash)
                seen_pixels.add(pixels_hash)
                signatures[path] = int(image_signature(path), 16)
                valid.append(path)
            except (OSError, ValueError) as error:
                rejected.append({"path": local_path(path), "reason": str(error)})
        if len(valid) < 15:
            raise ValueError(f"{folder.name}: only {len(valid)} unique decoded images; need at least 15.")
        near = [(left.name, right.name) for index, left in enumerate(valid)
                for right in valid[index + 1:]
                if (signatures[left] ^ signatures[right]).bit_count() <= 5]
        if near:
            warnings.append({"person": folder.name, "near_duplicate_pairs": near,
                             "warning": "Similar photos may not be independent captures. Random splits can overestimate generalization."})
        train_count = int(len(valid) * 0.6)
        validation_count = int(len(valid) * 0.2)
        for source, split in split_items(valid, train_count, validation_count):
            target = PROCESSED / "face" / split / user_id / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            records.append(sample_record(user_id, folder.name, "face", source, target,
                                          split, folder.name, source.name))
    write_json(MANIFESTS / "face_quality.json", {"rejected": rejected, "warnings": warnings,
               "independence": "File uniqueness is checked; capture-session independence and authenticity cannot be proved from these files."})
    return records


def speech_metadata(local_only=False):
    cache = RAW / "speech_metadata.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"speaker_id": str})
    if local_only:
        rows = []
        for word in (*COMMANDS, *UNSUPPORTED):
            for path in sorted((RAW / "speech_commands" / word.lower()).glob("*.wav")):
                match = re.fullmatch(r"(.+)_nohash_(\d+)\.wav", path.name)
                if match:
                    speaker, utterance = match.groups()
                    rows.append({"file": f"{word.lower()}/{path.name}", "label": word.lower(),
                                 "speaker_id": speaker, "utterance_id": int(utterance),
                                 "source_row": len(rows), "source_shard": "manual_archive",
                                 "official_split": "manual_archive_partition_not_recorded"})
        if not rows:
            raise FileNotFoundError("Extract genuine Speech Commands WAVs to data/raw/speech_commands/{on,off,go,stop,yes,no}/, then use --local-only.")
        metadata = pd.DataFrame(rows)
        metadata.to_csv(cache, index=False)
        return metadata
    info = get_json(f"https://huggingface.co/api/datasets/{VOICE_DATASET}/revision/{VOICE_REVISION}")
    paths = sorted(entry["rfilename"] for entry in info["siblings"]
                   if entry["rfilename"].startswith("data/train/")
                   and entry["rfilename"].endswith(".parquet"))
    if not paths:
        raise ValueError("Speech mirror did not expose the expected metadata shards.")
    frames, offset = [], 0
    for number, path in enumerate(paths, start=1):
        url = f"https://huggingface.co/datasets/{VOICE_DATASET}/resolve/{VOICE_REVISION}/{path}"
        with fsspec.open(url, "rb", block_size=65536) as stream:
            frame = pq.ParquetFile(stream).read(
                columns=["file", "label", "speaker_id", "utterance_id"]).to_pandas()
        frame["source_row"] = np.arange(offset, offset + len(frame))
        frame["source_shard"] = path
        offset += len(frame)
        frames.append(frame)
        print(f"Speech metadata {number}/{len(paths)}: {offset} rows", flush=True)
    metadata = pd.concat(frames, ignore_index=True)
    metadata.to_csv(cache, index=False)
    return metadata


def select_speakers(metadata):
    words = [word.lower() for word in (*COMMANDS, *UNSUPPORTED)]
    relevant = metadata[metadata["label"].isin(words)].drop_duplicates("file")
    counts = relevant.groupby(["speaker_id", "label"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=words, fill_value=0)
    candidates = sorted(counts.index[(counts >= 5).all(axis=1)].tolist())
    random.Random(SEED).shuffle(candidates)
    if len(candidates) < 3:
        raise ValueError(f"Only {len(candidates)} speakers have five genuine files for each of {words}. No audio was duplicated.")
    counts.loc[candidates].to_csv(MANIFESTS / "candidate_speaker_available_counts.csv")
    return candidates


def cache_utterances(selected_rows):
    for shard, choices in pd.DataFrame(selected_rows).groupby("source_shard", sort=True):
        missing = [row for row in choices.to_dict("records")
                   if not (RAW / "speech_commands" / row["file"]).exists()]
        if not missing:
            continue
        url = f"https://huggingface.co/datasets/{VOICE_DATASET}/resolve/{VOICE_REVISION}/{shard}"
        with fsspec.open(url, "rb", block_size=1024 * 1024) as stream:
            parquet = pq.ParquetFile(stream)
            start = 0
            for group_number in range(parquet.num_row_groups):
                end = start + parquet.metadata.row_group(group_number).num_rows
                needed = [row for row in missing if start <= row["shard_row"] < end]
                if needed:
                    rows = parquet.read_row_group(group_number, columns=["file", "audio"]).to_pylist()
                    for expected in needed:
                        actual = rows[int(expected["shard_row"]) - start]
                        if actual["file"] != expected["file"]:
                            raise ValueError("Pinned audio row does not match the selected source identifier.")
                        content = actual["audio"]["bytes"]
                        load_audio(content)
                        target = RAW / "speech_commands" / expected["file"]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(content)
                start = end
        print(f"Cached {len(missing)} selected utterances from {Path(shard).name}", flush=True)


def prepare_voice(people, download=True):
    metadata = speech_metadata(local_only=not download)
    metadata["shard_row"] = metadata.groupby("source_shard", sort=False).cumcount()
    candidates = select_speakers(metadata)
    records, selected, speakers, audit = [], [], [], []
    seen_content = set()
    for speaker in candidates:
        candidate = metadata[(metadata.speaker_id == speaker) & metadata.label.isin(
            [word.lower() for word in (*COMMANDS, *UNSUPPORTED)])].sort_values("file")
        candidate_rows = candidate.to_dict("records")
        if download:
            cache_utterances(candidate_rows)
        unique_words, duplicate_files = {}, []
        for word in (*COMMANDS, *UNSUPPORTED):
            unique, hashes = [], set(seen_content)
            for row in [row for row in candidate_rows if row["label"] == word.lower()]:
                path = RAW / "speech_commands" / row["file"]
                if not path.exists():
                    raise FileNotFoundError(f"Missing cached recording: {row['file']}")
                digest = decoded_hash(path, "voice")
                if digest in hashes:
                    duplicate_files.append(row["file"])
                    continue
                hashes.add(digest)
                unique.append(row)
            unique_words[word] = unique
        counts = {word: len(rows) for word, rows in unique_words.items()}
        accepted = min(counts.values()) >= 5
        audit.append({"speaker_id": speaker, "unique_counts": counts,
                      "duplicate_files": duplicate_files, "selected": accepted})
        if not accepted:
            print(f"Skip speaker {speaker}: insufficient distinct audio {counts}", flush=True)
            continue
        user_id, folder = list(people.items())[len(speakers)]
        speakers.append(speaker)
        for word, choices in unique_words.items():
            random.Random(SEED).shuffle(choices)
            for row, split in split_items(choices[:5], 3, 1):
                seen_content.add(decoded_hash(RAW / "speech_commands" / row["file"], "voice"))
                selected.append({**row, "user_id": user_id, "display_name": folder.name,
                                 "command": word, "split": split})
        if len(speakers) == 3:
            break
    write_json(MANIFESTS / "voice_selection_audit.json", audit)
    if len(speakers) != 3:
        raise ValueError("Could not find three speakers with five distinct decoded utterances per word.")
    pd.DataFrame([{"speaker_id": entry["speaker_id"], **entry["unique_counts"]}
                  for entry in audit if entry["selected"]]).to_csv(
                      MANIFESTS / "selected_speaker_available_counts.csv", index=False)
    for row in selected:
        user_id, split, word = row["user_id"], row["split"], row["command"]
        source = RAW / "speech_commands" / row["file"]
        target = PROCESSED / "voice" / split / user_id / f"{word}_{source.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, load_audio(source), 16000, subtype="PCM_16")
        records.append(sample_record(user_id, row["display_name"], "voice", source, target,
                                      split, row["speaker_id"], row["file"], word,
                                      row.get("official_split", "official_train")))
    print(f"Voice: {len(records)} real utterances from speakers {speakers}", flush=True)
    return records


def fingerprint_archive(download=True):
    target = RAW / "DB1_B.zip"
    if target.exists():
        if not zipfile.is_zipfile(target):
            raise ValueError(f"Not a valid fingerprint ZIP: {target}")
        return target
    if not download:
        raise FileNotFoundError("Place the official FVC2004 DB1_B.zip in data/raw/DB1_B.zip.")
    attempts = []
    for url in (FVC_PAGE, FVC_PAGE.replace("https:", "http:")):
        try:
            response = requests.get(url, timeout=40)
            response.raise_for_status()
            attempts.append({"url": url, "status": response.status_code})
            if "DB1_B.zip" in response.text:
                break
        except requests.RequestException as error:
            attempts.append({"url": url, "error": str(error)})
    response = requests.get(FVC_ZIP, timeout=90)
    response.raise_for_status()
    if not zipfile.is_zipfile(io.BytesIO(response.content)):
        raise ValueError("Official fingerprint download did not contain a ZIP archive.")
    target.write_bytes(response.content)
    write_json(MANIFESTS / "fingerprint_download.json", {"source": FVC_ZIP,
               "sha256": file_hash(target), "page_attempts": attempts,
               "transport": "Official site advertises HTTP. No TLS checks were disabled.",
               "terms": "Public official sample for local academic use; no redistribution license asserted. Do not redistribute biometric files."})
    return target


def prepare_fingerprints(people, download=True):
    archive_path = fingerprint_archive(download)
    records = []
    with zipfile.ZipFile(archive_path) as archive:
        members = {}
        for name in archive.namelist():
            match = re.fullmatch(r"(\d+)_(\d+)\.(?:tif|tiff|bmp|png)", Path(name).name, re.I)
            if match:
                finger, impression = match.groups()
                members.setdefault(finger, []).append((int(impression), name))
        selected = sorted(finger for finger, files in members.items()
                          if len(files) == 8 and len({number for number, _ in files}) == 8)[:3]
        if len(selected) != 3:
            raise ValueError("DB1_B must contain three real fingers with eight distinct impressions each.")
        for (user_id, folder), finger in zip(people.items(), selected):
            seen = set()
            for (impression, member), split in split_items(sorted(members[finger]), 5, 1):
                source = RAW / "fingerprints" / Path(member).name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(archive.read(member))
                content_hash = decoded_hash(source, "fingerprint")
                if content_hash in seen:
                    raise ValueError(f"Duplicated fingerprint impression in finger {finger}.")
                seen.add(content_hash)
                target = PROCESSED / "fingerprint" / split / user_id / f"{finger}_{impression}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(source) as image:
                    ImageOps.exif_transpose(image).convert("L").save(target)
                records.append(sample_record(user_id, folder.name, "fingerprint", member,
                                              target, split, f"FVC2004_DB1_B:{finger}",
                                              f"{finger}_{impression}", source_split="official_B"))
    return records


def validate_manifest(frame):
    if frame.empty:
        raise ValueError("No real data is available.")
    if frame.sample_id.duplicated().any():
        raise ValueError("Repeated source identifiers in sample manifest.")
    for modality, group in frame.groupby("modality"):
        for column in ("sha256", "decoded_sha256"):
            if group.groupby(column)["split"].nunique().gt(1).any():
                raise ValueError(f"{modality}: duplicated {column} across splits.")
        if modality != "face" and group.groupby("user_id").source_identity.nunique().gt(1).any():
            raise ValueError(f"{modality}: more than one source identity mapped to a user.")
        if group.groupby("source_identity").user_id.nunique().gt(1).any():
            raise ValueError(f"{modality}: one source identity assigned to multiple users.")
    for row in frame.to_dict("records"):
        if not (ROOT / row["path"]).is_file() or file_hash(ROOT / row["path"]) != row["sha256"]:
            raise ValueError(f"Missing or altered prepared sample: {row['path']}")
    return True


def build_fusion_manifests(frame):
    pairs, triplets = [], []
    demo_words = {"user_01": "ON", "user_02": "OFF", "user_03": "STOP"}
    for user_id in sorted(frame.user_id.unique()):
        for split in ("train", "validation", "test"):
            subset = frame[(frame.user_id == user_id) & (frame.split == split)]
            faces = subset[subset.modality == "face"].sort_values("sample_id").to_dict("records")
            voices = subset[subset.modality == "voice"].sort_values("sample_id").to_dict("records")
            fingers = subset[subset.modality == "fingerprint"].sort_values("sample_id").to_dict("records")
            if split == "test":
                voices.sort(key=lambda row: (row["command"] != demo_words[user_id], row["sample_id"]))
            local_pairs = []
            for number, (face, voice) in enumerate(zip(faces, voices)):
                pair = {"pair_id": f"fv:{user_id}:{split}:{number}", "user_id": user_id,
                        "split": split, "face_id": face["sample_id"], "voice_id": voice["sample_id"]}
                pairs.append(pair)
                local_pairs.append(pair)
            for number, (pair, finger) in enumerate(zip(local_pairs, fingers)):
                triplets.append({**pair, "triplet_id": f"fvf:{user_id}:{split}:{number}",
                                 "fingerprint_id": finger["sample_id"]})
    pair_frame = pd.DataFrame(pairs, columns=["pair_id", "user_id", "split", "face_id", "voice_id"])
    triplet_frame = pd.DataFrame(triplets, columns=["pair_id", "user_id", "split", "face_id", "voice_id", "triplet_id", "fingerprint_id"])
    pair_frame.to_csv(MANIFESTS / "face_voice_pairs.csv", index=False)
    triplet_frame.to_csv(MANIFESTS / "three_modality_triplets.csv", index=False)
    return pair_frame, triplet_frame


def create_live_test(frame, triplets):
    if triplets.empty:
        return []
    lookup = frame.set_index("sample_id").to_dict("index")
    records = []
    for user_id, attempts in triplets[triplets.split == "test"].groupby("user_id"):
        attempt = attempts.iloc[0]
        face, voice, finger = [lookup[attempt[column]]
                                for column in ("face_id", "voice_id", "fingerprint_id")]
        folder = LIVE / face["display_name"]
        folder.mkdir(parents=True, exist_ok=True)
        destinations = (folder / "face.jpg", folder / "voice.wav", folder / "fingerprint.png")
        with Image.open(ROOT / face["path"]) as image:
            image.convert("RGB").save(destinations[0], quality=95)
        shutil.copyfile(ROOT / voice["path"], destinations[1])
        shutil.copyfile(ROOT / finger["path"], destinations[2])
        records.append({"user_id": user_id, "display_name": face["display_name"],
                        "face_path": local_path(destinations[0]), "voice_path": local_path(destinations[1]),
                        "fingerprint_path": local_path(destinations[2]),
                        "original_speaker_id": voice["source_identity"],
                        "original_finger_id": finger["source_identity"], "command_label": voice["command"],
                        "source_split": "test", "triplet_id": attempt["triplet_id"],
                        "source_identifiers": json.dumps([face["source_identifier"], voice["source_identifier"], finger["source_identifier"]]),
                        "sample_ids": json.dumps([attempt["face_id"], attempt["voice_id"], attempt["fingerprint_id"]]),
                        "file_hashes": json.dumps([file_hash(path) for path in destinations])})
    pd.DataFrame(records).to_csv(LIVE / "manifest.csv", index=False)
    return records


def setup(download=True, faces_only=False):
    make_folders()
    people = discover_people()
    records = prepare_faces(people)
    blocked = {}
    for modality, prepare in (("voice", prepare_voice), ("fingerprint", prepare_fingerprints)):
        if faces_only:
            blocked[modality] = "Not acquired in --faces-only mode."
            continue
        try:
            records.extend(prepare(people, download=download))
        except (requests.RequestException, OSError, ValueError, KeyError) as error:
            blocked[modality] = f"{type(error).__name__}: {error}"
            print(f"BLOCKED {modality}: {error}", flush=True)
    frame = pd.DataFrame(records)
    validate_manifest(frame)
    frame.to_csv(MANIFESTS / "samples.csv", index=False)
    mapping = {}
    for user_id, folder in people.items():
        modalities = frame[frame.user_id == user_id]
        voice = modalities[modalities.modality == "voice"]
        finger = modalities[modalities.modality == "fingerprint"]
        mapping[user_id] = {"display_name": folder.name, "role": ROLES[user_id],
                            "face_identity": folder.name,
                            "voice_identity": None if voice.empty else voice.iloc[0].source_identity,
                            "fingerprint_identity": None if finger.empty else finger.iloc[0].source_identity,
                            "virtual_identity": True}
    write_json(MANIFESTS / "identities.json", mapping)
    pairs, triplets = build_fusion_manifests(frame)
    live = create_live_test(frame, triplets)
    summary = {"status": "BLOCKED" if blocked else "PREPARED", "blocked": blocked,
               "counts": frame.groupby(["modality", "user_id", "split"]).size().to_string(),
               "face_voice_pairs": len(pairs), "three_modality_triplets": len(triplets),
               "live_test_people": len(live), "seed": SEED, "voice_revision": VOICE_REVISION,
               "warning": "Educational virtual identities; modalities are from unrelated physical people. Local utterance splits are not official benchmark splits."}
    write_json(MANIFESTS / "setup_status.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-only", action="store_true", help="Use only cached authentic downloads.")
    parser.add_argument("--faces-only", action="store_true", help="Prepare faces and explicitly block other modalities.")
    arguments = parser.parse_args()
    status = setup(download=not arguments.local_only, faces_only=arguments.faces_only)
    raise SystemExit(0 if status["status"] == "PREPARED" else 2)