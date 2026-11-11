"""Train exactly six tasks, each with KNN(3) and a linear SVM."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from commands import calibrate_command_threshold
from config import MANIFESTS, MODELS, PROCESSED, RESULTS, ROOT, SEED, make_folders
from dataset_setup import file_hash, local_path, validate_manifest, write_json
from fusion import make_fusion_matrix
from verification import EXTRACTORS, calibrate_threshold, transform_features


def prepare_feature_data(frame):
    preprocessors, vectors, dimensions = {}, {}, {}
    for modality, extractor in EXTRACTORS.items():
        subset = frame[frame.modality == modality]
        if subset.empty:
            continue
        raw = np.asarray([extractor(ROOT / path) for path in subset.path], dtype=np.float32)
        training_mask = subset.split.eq("train").to_numpy()
        preprocessor = make_pipeline(StandardScaler(), PCA(n_components=0.95,
                                      svd_solver="full", whiten=True, random_state=SEED))
        preprocessor.fit(raw[training_mask])
        transformed = np.asarray([transform_features(preprocessor, row) for row in raw], dtype=np.float32)
        preprocessors[modality] = preprocessor
        vectors.update(zip(subset.sample_id, transformed))
        dimensions[modality] = {"raw": raw.shape[1], "after_pca": transformed.shape[1],
                                "pca_fit_samples": int(training_mask.sum()),
                                "variance_preserved": float(preprocessor[-1].explained_variance_ratio_.sum())}
        print(f"{modality}: {raw.shape[1]} raw -> {transformed.shape[1]} PCA features", flush=True)
    return preprocessors, vectors, dimensions


def task_data(frame, vectors):
    tasks = {}
    for modality in EXTRACTORS:
        subset = frame[frame.modality == modality]
        if not subset.empty:
            tasks[modality] = {split: (
                np.asarray([vectors[value] for value in group.sample_id]),
                group.user_id.to_numpy(), group.sample_id.tolist())
                for split, group in subset.groupby("split")}
    voice = frame[frame.modality == "voice"]
    if not voice.empty:
        voice = voice.copy()
        voice["command_target"] = voice.command.where(voice.command.isin(["ON", "OFF", "GO", "STOP"]), "UNKNOWN")
        tasks["command"] = {split: (
            np.asarray([vectors[value] for value in group.sample_id]),
            group.command_target.to_numpy(), group.sample_id.tolist())
            for split, group in voice.groupby("split")}
    for task, name, modalities in (("face_voice", "face_voice_pairs.csv", ("face", "voice")),
                                    ("fusion3", "three_modality_triplets.csv", ("face", "voice", "fingerprint"))):
        path = MANIFESTS / name
        if not path.exists():
            continue
        manifest = pd.read_csv(path)
        if not manifest.empty:
            tasks[task] = {}
            for split, group in manifest.groupby("split"):
                values, labels = make_fusion_matrix(group, vectors, modalities)
                identifier = "triplet_id" if task == "fusion3" else "pair_id"
                tasks[task][split] = (values, labels, group[identifier].tolist())
    return tasks


def train_models():
    make_folders()
    frame = pd.read_csv(MANIFESTS / "samples.csv", keep_default_na=False)
    validate_manifest(frame)
    preprocessors, vectors, dimensions = prepare_feature_data(frame)
    joblib.dump(preprocessors, MODELS / "preprocessors.joblib")
    np.savez_compressed(PROCESSED / "features.npz", ids=np.asarray(list(vectors)),
                        **{f"sample_{number}": value for number, value in enumerate(vectors.values())})
    tasks = task_data(frame, vectors)
    results, model_paths, selected = [], {}, {}
    for task, splits in tasks.items():
        training, training_labels, training_ids = splits["train"]
        validation, validation_labels, validation_ids = splits["validation"]
        model_paths[task] = {}
        for algorithm, classifier in (("knn", KNeighborsClassifier(n_neighbors=3)),
                                      ("svm", SVC(kernel="linear"))):
            classifier.fit(training, training_labels)
            predicted = classifier.predict(validation)
            calibrate = calibrate_command_threshold if task == "command" else calibrate_threshold
            calibration = calibrate(classifier, training, training_labels, validation, validation_labels)
            validation_trials = calibration.pop("validation_trials")
            pd.DataFrame(validation_trials).to_csv(
                RESULTS / "metrics" / f"{task}_{algorithm}_validation_threshold_trials.csv", index=False)
            artifact = {"task": task, "algorithm": algorithm, "classifier": classifier,
                        "n_features": training.shape[1], "training_ids": training_ids,
                        "validation_ids": validation_ids, **calibration}
            path = MODELS / f"{task}_{algorithm}.joblib"
            joblib.dump(artifact, path)
            model_paths[task][algorithm] = local_path(path)
            results.append({"task": task, "algorithm": algorithm,
                            "validation_accuracy": accuracy_score(validation_labels, predicted),
                            "validation_macro_f1": f1_score(validation_labels, predicted, average="macro", zero_division=0),
                            "validation_far": calibration["validation_far"],
                            "validation_frr": calibration["validation_frr"],
                            "threshold": calibration["threshold"], "train_samples": len(training),
                            "validation_samples": len(validation), "feature_count": training.shape[1]})
            print(f"Trained {task}_{algorithm}: validation accuracy {results[-1]['validation_accuracy']:.3f}", flush=True)
        comparison = [row for row in results if row["task"] == task]
        winner = max(comparison, key=lambda row: (row["validation_macro_f1"], row["validation_accuracy"], row["algorithm"] == "knn"))
        selected[task] = winner["algorithm"]
    pd.DataFrame(results).to_csv(RESULTS / "metrics" / "model_comparison.csv", index=False)
    expected = {"face", "voice", "fingerprint", "face_voice", "fusion3", "command"}
    source_files = ["features.py", "config.py", "fusion.py", "commands.py", "verification.py", "train.py",
                    "data/manifests/samples.csv",
                    "data/manifests/identities.json", "data/manifests/face_voice_pairs.csv",
                    "data/manifests/three_modality_triplets.csv"]
    artifact_files = [MODELS / "preprocessors.joblib", *[ROOT / value for paths in model_paths.values() for value in paths.values()]]
    index = {"selected": selected, "models": model_paths, "dimensions": dimensions,
             "status": "TRAINED" if set(tasks) == expected else "PARTIAL",
             "missing_tasks": sorted(expected - set(tasks)), "seed": SEED,
             "source_hashes": {name: file_hash(ROOT / name) for name in source_files},
             "artifact_hashes": {local_path(path): file_hash(path) for path in artifact_files},
             "selection_rule": "Highest validation macro-F1, then validation accuracy, then KNN on exact ties. No test tuning.",
             "threshold_rule": "Minimize validation FAR + FRR; ties prefer lower FAR, then a smaller threshold.",
             "command_threshold_rule": "Use actual supported prediction AND distance rule on supported/UNKNOWN validation words, not claimed-identity trials.",
             "fusion_rule": "Concatenate train-only scaled/PCA-whitened blocks; divide each block by sqrt(its dimension). No repeated source within a fusion split."}
    write_json(MODELS / "model_index.json", index)
    print(json.dumps({"status": index["status"], "selected": selected, "classifiers": len(results)}, indent=2))
    return index


if __name__ == "__main__":
    outcome = train_models()
    raise SystemExit(0 if outcome["status"] == "TRAINED" else 2)