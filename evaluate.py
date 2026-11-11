"""Report frozen-model test results, never tune models using these outcomes."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

from commands import recognize_command
from config import COMMANDS, LIVE, MANIFESTS, RESULTS, ROOT
from dataset_setup import validate_manifest, write_json
from permissions import check_permission
from pipeline import run_pipeline
from train import task_data
from verification import extract_vector, load_system, verify_vector


def classification_metrics(actual, predicted):
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(actual, predicted)),
            "precision_macro": float(precision), "recall_macro": float(recall),
            "f1_macro": float(f1)}


def verification_metrics(genuine, accepted):
    genuine, accepted = np.asarray(genuine, dtype=bool), np.asarray(accepted, dtype=bool)
    impostor_count, genuine_count = int((~genuine).sum()), int(genuine.sum())
    false_accepts = int((~genuine & accepted).sum())
    false_rejects = int((genuine & ~accepted).sum())
    precision, recall, f1, _ = precision_recall_fscore_support(
        genuine, accepted, average="binary", zero_division=0)
    return {"far": false_accepts / impostor_count if impostor_count else None,
            "frr": false_rejects / genuine_count if genuine_count else None,
            "false_accepts": false_accepts, "impostor_attempts": impostor_count,
            "false_rejects": false_rejects, "genuine_attempts": genuine_count,
            "verification_precision": float(precision), "verification_recall": float(recall),
            "verification_f1": float(f1),
            "verification_accuracy": float((genuine == accepted).mean())}


def save_confusion(actual, predicted, labels, name):
    matrix = confusion_matrix(actual, predicted, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        RESULTS / "confusion_matrices" / f"{name}.csv")
    figure, axis = plt.subplots(figsize=(6, 4.5))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        ax=axis, colorbar=False, cmap="Greens", values_format="d")
    axis.set_title(name.replace("_", " "))
    figure.tight_layout()
    figure.savefig(RESULTS / "confusion_matrices" / f"{name}.png", dpi=140)
    plt.close(figure)


def action_metrics(attempts):
    expected = attempts.expected_command.fillna("NO_ACTION")
    executed = attempts.executed_command.fillna("NO_ACTION")
    return {"action_execution_decision_accuracy": float((attempts.expected_action == attempts.action_executed).mean()),
            "exact_action_accuracy": float((expected == executed).mean()),
            "executed_actions": int(attempts.action_executed.sum()),
            "unexpected_actions_vs_source_word": int((attempts.action_executed & (expected != executed)).sum()),
            "wrong_executed_commands": int((attempts.action_executed &
                                            (attempts.source_command != attempts.executed_command)).sum()),
            "permission_violations_for_recognized_word": int(attempts.permission_violation.sum())}


def evaluate_models(system, tasks):
    rows = []
    for task, splits in tasks.items():
        values, actual, identifiers = splits["test"]
        for algorithm, artifact in system["models"][task].items():
            predicted = artifact["classifier"].predict(values)
            row = {"task": task, "algorithm": algorithm,
                   "selected": system["selected"][task] == algorithm,
                   "independent_test_observations": len(values),
                   **classification_metrics(actual, predicted)}
            labels = sorted(set(actual) | set(predicted))
            save_confusion(actual, predicted, labels, f"{task}_{algorithm}_classification")
            if task == "command":
                decisions = [recognize_command(vector, system, algorithm) for vector in values]
                recognized = [decision["command"] for decision in decisions]
                supported = np.isin(actual, COMMANDS)
                accepted = np.asarray([decision["supported"] for decision in decisions])
                row.update(verification_metrics(supported, accepted))
                row.update({f"recognition_{key}": value for key, value in
                            classification_metrics(actual, recognized).items()})
                row["rate_meaning"] = "Command FAR: unsupported YES/NO accepted as supported; FRR: supported word rejected as UNKNOWN. Not biometric FAR/FRR."
                pd.DataFrame({"sample_id": identifiers, "actual": actual,
                              "raw_prediction": predicted, "recognized": recognized}).to_csv(
                                  RESULTS / "metrics" / f"command_{algorithm}_test_predictions.csv", index=False)
                save_confusion(actual, recognized, sorted(set(actual) | set(recognized)),
                               f"command_{algorithm}_with_rejection")
            else:
                attempts = []
                for vector, user_id, identifier in zip(values, actual, identifiers):
                    for claim in system["identities"]:
                        decision = verify_vector(task, vector, claim, system, algorithm)
                        attempts.append({"sample_id": identifier, "actual_identity": user_id,
                                         "genuine": bool(user_id == claim), **decision})
                attempts = pd.DataFrame(attempts)
                row.update(verification_metrics(attempts.genuine, attempts.accepted))
                row["rate_meaning"] = "All enrolled genuine claims and both wrong enrolled claims per held-out observation. Not open-set security certification."
                attempts.to_csv(RESULTS / "metrics" / f"{task}_{algorithm}_test_verification.csv", index=False)
                save_confusion(attempts.genuine, attempts.accepted, [False, True],
                               f"{task}_{algorithm}_verification")
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "metrics" / "test_metrics.csv", index=False)
    return frame


def evaluate_hierarchy(system, samples):
    triplets = pd.read_csv(MANIFESTS / "three_modality_triplets.csv")
    triplets = triplets[triplets.split == "test"]
    lookup = samples.set_index("sample_id").to_dict("index")
    summaries, traces = [], []
    for variant in ("knn", "svm", "selected"):
        current = system if variant == "selected" else {**system, "selected": {
            task: variant for task in system["models"]}}
        attempts = []
        for triplet in triplets.to_dict("records"):
            face, voice, finger = [lookup[triplet[name]] for name in ("face_id", "voice_id", "fingerprint_id")]
            for claim in system["identities"]:
                result = run_pipeline(ROOT / face["path"], ROOT / voice["path"],
                                      ROOT / finger["path"], claim, current)
                genuine = triplet["user_id"] == claim
                expected_action = genuine and voice["command"] in COMMANDS and check_permission(claim, voice["command"]) == "ALLOWED"
                attempts.append({"variant": variant, "triplet_id": triplet["triplet_id"],
                                 "source_identity": triplet["user_id"], "claimed_identity": claim,
                                 "genuine": genuine, "authenticated": result["authenticated"],
                                 "source_command": voice["command"], "recognized_command":
                                 None if result["command"] is None else result["command"]["command"],
                                 "expected_action": expected_action, "action_executed": result["action"] is not None,
                                 "expected_command": voice["command"] if expected_action else None,
                                 "executed_command": None if result["action"] is None else result["command"]["command"],
                                 "permission_violation": result["action"] is not None and (
                                     result["command"] is None or check_permission(
                                         claim, result["command"]["command"]) != "ALLOWED"),
                                 "result": result["result"]})
                traces.append({"variant": variant, "triplet_id": triplet["triplet_id"], **result})
        frame = pd.DataFrame(attempts)
        if frame.empty:
            continue
        frame.to_csv(RESULTS / "metrics" / f"hierarchy_{variant}_test_attempts.csv", index=False)
        summary = {"variant": variant, "independent_test_triplets": len(triplets),
                   "claim_attempts": len(frame), **classification_metrics(frame.genuine, frame.authenticated),
                   **verification_metrics(frame.genuine, frame.authenticated),
                   **action_metrics(frame)}
        summaries.append(summary)
        save_confusion(frame.genuine, frame.authenticated, [False, True], f"hierarchy_{variant}_verification")
        save_confusion(frame.expected_action, frame.action_executed, [False, True], f"hierarchy_{variant}_actions")
    pd.DataFrame(summaries).to_csv(RESULTS / "metrics" / "hierarchy_metrics.csv", index=False)
    write_json(RESULTS / "metrics" / "hierarchy_traces.json", traces)
    return summaries


def evaluate_demo(system):
    manifest_path = LIVE / "manifest.csv"
    if not manifest_path.exists():
        return []
    rows = []
    for sample in pd.read_csv(manifest_path).to_dict("records"):
        result = run_pipeline(ROOT / sample["face_path"], ROOT / sample["voice_path"],
                              ROOT / sample["fingerprint_path"], sample["user_id"], system)
        rows.append({"display_name": sample["display_name"],
                     "note": "Held-out test fixture, re-encoded face JPEG. Not an additional independent observation.", **result})
    write_json(RESULTS / "metrics" / "demo_traces.json", rows)
    return rows


def evaluate():
    system = load_system()
    samples = pd.read_csv(MANIFESTS / "samples.csv", keep_default_na=False)
    validate_manifest(samples)
    vectors = {row.sample_id: extract_vector(row.modality, ROOT / row.path, system)
               for row in samples.itertuples()}
    tasks = task_data(samples, vectors)
    metrics = evaluate_models(system, tasks)
    hierarchy = evaluate_hierarchy(system, samples)
    demos = evaluate_demo(system)
    report = {"selected_models": system["selected"],
              "selected_test_metrics": json.loads(metrics[metrics.selected].to_json(orient="records")),
              "hierarchical_test_results": hierarchy,
              "demo_results": [{"person": row["display_name"], "result": row["result"]} for row in demos],
              "independence": "Different claims on one observation are dependent trials. Demo copies are not extra test observations. Fusion pairing is virtual, not physical identity evidence.",
              "webcam": "Implemented separately in Streamlit; real camera demonstration must be established on available hardware."}
    write_json(RESULTS / "metrics" / "evaluation_summary.json", report)
    print(metrics[metrics.selected][["task", "algorithm", "accuracy", "f1_macro", "far", "frr"]].to_string(index=False))
    print(pd.DataFrame(hierarchy).to_string(index=False))
    print("Prepared demo results:", report["demo_results"])
    return report


if __name__ == "__main__":
    evaluate()