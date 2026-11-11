"""Early fusion combines feature values, never predicted class labels."""

import numpy as np


def concatenate_features(*vectors):
    arrays = [np.asarray(vector, dtype=np.float32) for vector in vectors]
    if not arrays or any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Fusion needs finite face, voice and/or fingerprint features.")
    return np.concatenate(arrays, axis=-1)


def make_fusion_matrix(manifest, vectors, modalities):
    features, labels = [], []
    for row in manifest.to_dict("records"):
        features.append(concatenate_features(
            *[vectors[row[f"{modality}_id"]] for modality in modalities]))
        labels.append(row["user_id"])
    return np.asarray(features, dtype=np.float32), np.asarray(labels)