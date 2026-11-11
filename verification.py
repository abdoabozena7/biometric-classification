"""Classification plus a validation-calibrated claimed-identity distance check."""

import json

import joblib
import numpy as np

from config import MANIFESTS, MODELS, ROOT
from dataset_setup import file_hash
from features import face_features, fingerprint_features, voice_features
from fusion import concatenate_features

EXTRACTORS = {"face": face_features, "voice": voice_features,
              "fingerprint": fingerprint_features}


def transform_features(preprocessor, raw_features):
    values = np.asarray(raw_features, dtype=np.float32)
    single_sample = values.ndim == 1
    reduced = preprocessor.transform(np.atleast_2d(values))
    reduced = (reduced / np.sqrt(reduced.shape[1])).astype(np.float32)
    if not np.isfinite(reduced).all():
        raise ValueError("Preprocessing produced non-finite features.")
    return reduced[0] if single_sample else reduced


def extract_vector(modality, source, system):
    return transform_features(system["preprocessors"][modality], EXTRACTORS[modality](source))


def calibrate_threshold(classifier, training, training_labels, validation, validation_labels):
    centroids = {label: training[training_labels == label].mean(axis=0)
                 for label in np.unique(training_labels)}
    predictions = classifier.predict(validation)
    trials = []
    for index, (vector, actual, predicted) in enumerate(zip(validation, validation_labels, predictions)):
        for claim, centroid in centroids.items():
            trials.append({"validation_row": index, "actual": str(actual), "claimed": str(claim),
                           "predicted": str(predicted), "genuine": bool(actual == claim),
                           "distance": float(np.linalg.norm(vector - centroid))})
    candidates = sorted({0.0, *[row["distance"] for row in trials]})
    best = None
    for threshold in candidates:
        genuine = [row for row in trials if row["genuine"]]
        impostors = [row for row in trials if not row["genuine"]]
        accepts = lambda row: row["predicted"] == row["claimed"] and row["distance"] <= threshold
        far = sum(accepts(row) for row in impostors) / len(impostors)
        frr = sum(not accepts(row) for row in genuine) / len(genuine)
        score = (far + frr, far, threshold)
        if best is None or score < best[0]:
            best = (score, threshold, far, frr)
    return {"centroids": centroids, "threshold": best[1],
            "validation_far": best[2], "validation_frr": best[3], "validation_trials": trials}


def load_system():
    index_path = MODELS / "model_index.json"
    if not index_path.exists():
        raise FileNotFoundError("Trained models are missing. Run dataset_setup.py, then train.py.")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for relative, expected in index["artifact_hashes"].items():
        path = ROOT / relative
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Model or preprocessing artifact is missing/changed: {relative}. Retrain locally.")
    for relative, expected in index["source_hashes"].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError(f"Training data/feature configuration changed: {relative}. Run train.py again.")
    identities = json.loads((MANIFESTS / "identities.json").read_text(encoding="utf-8"))
    system = {"preprocessors": joblib.load(MODELS / "preprocessors.joblib"),
              "models": {}, "selected": index["selected"], "identities": identities,
              "index": index}
    for task, algorithms in index["models"].items():
        system["models"][task] = {algorithm: joblib.load(ROOT / relative)
                                  for algorithm, relative in algorithms.items()}
    return system


def verify_vector(task, vector, claimed_identity, system, algorithm=None):
    result = {"accepted": False, "claimed_identity": claimed_identity,
              "predicted_identity": None, "distance": None, "threshold": None,
              "explanation": "Verification did not run."}
    try:
        if claimed_identity not in system["identities"]:
            raise ValueError("Claimed identity is not enrolled.")
        algorithm = algorithm or system["selected"][task]
        artifact = system["models"][task][algorithm]
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (artifact["n_features"],) or not np.isfinite(vector).all():
            raise ValueError("Unsupported feature vector shape or non-finite values.")
        predicted = str(artifact["classifier"].predict(vector.reshape(1, -1))[0])
        distance = float(np.linalg.norm(vector - artifact["centroids"][claimed_identity]))
        threshold = float(artifact["threshold"])
        accepted = predicted == claimed_identity and distance <= threshold
        result.update(accepted=bool(accepted), predicted_identity=predicted,
                      distance=distance, threshold=threshold, algorithm=algorithm,
                      explanation="Prediction matches the claim and distance is within the validation threshold."
                      if accepted else "Identity prediction disagrees with the claim or distance exceeds the validation threshold.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["explanation"] = str(error)
    return result


def verify_sources(task, sources, claimed_identity, system=None):
    try:
        system = system or load_system()
        vectors = [extract_vector(modality, source, system) for modality, source in sources]
        vector = vectors[0] if len(vectors) == 1 else concatenate_features(*vectors)
        return verify_vector(task, vector, claimed_identity, system)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        return {"accepted": False, "claimed_identity": claimed_identity,
                "predicted_identity": None, "distance": None, "threshold": None,
                "explanation": str(error)}


def verify_face(source, claimed_identity, system=None):
    return verify_sources("face", [("face", source)], claimed_identity, system)


def verify_voice(source, claimed_identity, system=None):
    return verify_sources("voice", [("voice", source)], claimed_identity, system)


def verify_fingerprint(source, claimed_identity, system=None):
    return verify_sources("fingerprint", [("fingerprint", source)], claimed_identity, system)


def verify_face_voice_fusion(face, voice, claimed_identity, system=None):
    return verify_sources("face_voice", [("face", face), ("voice", voice)], claimed_identity, system)


def verify_three_modality_fusion(face, voice, fingerprint, claimed_identity, system=None):
    return verify_sources("fusion3", [("face", face), ("voice", voice),
                                      ("fingerprint", fingerprint)], claimed_identity, system)