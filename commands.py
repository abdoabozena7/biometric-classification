"""Recognize an isolated word from real traditional audio features."""

import numpy as np

from config import COMMANDS


def calibrate_command_threshold(classifier, training, training_labels,
                                 validation, validation_labels):
    centroids = {label: training[training_labels == label].mean(axis=0)
                 for label in np.unique(training_labels)}
    predicted = classifier.predict(validation)
    distances = np.asarray([np.linalg.norm(vector - centroids[label])
                            for vector, label in zip(validation, predicted)])
    supported_truth = np.isin(validation_labels, COMMANDS)
    supported_prediction = np.isin(predicted, COMMANDS)
    best = None
    for threshold in sorted({0.0, *distances.tolist()}):
        accepted = supported_prediction & (distances <= threshold)
        far = float(accepted[~supported_truth].mean())
        frr = float((~accepted[supported_truth]).mean())
        score = (far + frr, far, threshold)
        if best is None or score < best[0]:
            best = (score, threshold, far, frr)
    trials = [{"validation_row": number, "actual_word": str(actual),
               "predicted_word": str(label), "supported_truth": bool(supported),
               "distance": float(distance)}
              for number, (actual, label, supported, distance) in enumerate(
                  zip(validation_labels, predicted, supported_truth, distances))]
    return {"centroids": centroids, "threshold": best[1], "validation_far": best[2],
            "validation_frr": best[3], "validation_trials": trials}


def recognize_command(voice_vector, system, algorithm=None):
    algorithm = algorithm or system["selected"]["command"]
    artifact = system["models"]["command"][algorithm]
    vector = np.asarray(voice_vector, dtype=np.float32)
    if vector.shape != (artifact["n_features"],) or not np.isfinite(vector).all():
        raise ValueError("Invalid command audio features.")
    predicted = str(artifact["classifier"].predict(vector.reshape(1, -1))[0])
    distance = float(np.linalg.norm(vector - artifact["centroids"][predicted]))
    supported = predicted in COMMANDS and distance <= artifact["threshold"]
    return {"command": predicted if supported else "UNKNOWN", "predicted_label": predicted,
            "distance": distance, "threshold": float(artifact["threshold"]),
            "algorithm": algorithm, "supported": bool(supported)}


def execute_action(command, state):
    if command not in COMMANDS:
        raise ValueError("An unsupported command cannot execute.")
    if command == "ON":
        state["enabled"] = True
        return "Simulated protected system enabled."
    if command == "OFF":
        state["enabled"] = False
        return "Simulated protected system disabled."
    if command == "GO":
        state["running"] = True
        return "Simulated process started."
    state["running"] = False
    return "Simulated process stopped."