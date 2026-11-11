"""Doubles here test routing only; trained-model tests below use real artifacts."""

import io
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import pipeline
from config import LIVE, MANIFESTS, PROCESSED, ROOT
from dataset_setup import file_hash, validate_manifest
from features import face_features, fingerprint_features, voice_features
from fusion import concatenate_features
from permissions import check_permission
from verification import load_system, verify_face, verify_vector


@pytest.mark.parametrize("user,command,expected", [
    ("user_01", "ON", "ALLOWED"), ("user_02", "ON", "DENIED"),
    ("user_03", "GO", "DENIED"), ("user_03", "STOP", "ALLOWED"),
    ("user_01", "UNKNOWN", "DENIED"), ("not_enrolled", "OFF", "DENIED")])
def test_permissions(user, command, expected):
    assert check_permission(user, command) == expected


def routing_doubles(monkeypatch, fail_stage=None, word="ON"):
    calls = []

    def extract(modality, source, system):
        calls.append(f"extract:{modality}")
        return np.ones(3, dtype=np.float32)

    def verify(task, vector, claim, system):
        calls.append(task)
        return {"accepted": task != fail_stage, "claimed_identity": claim,
                "predicted_identity": claim, "distance": 0.1, "threshold": 0.2,
                "explanation": "TEST DOUBLE: routing test only."}

    def recognize(vector, system):
        calls.append("command")
        return {"command": word, "supported": word != "UNKNOWN"}

    monkeypatch.setattr(pipeline, "extract_vector", extract)
    monkeypatch.setattr(pipeline, "verify_vector", verify)
    monkeypatch.setattr(pipeline, "recognize_command", recognize)
    return calls, {"identities": {user: {} for user in ("user_01", "user_02", "user_03")}}


@pytest.mark.parametrize("user,word,expected", [
    ("user_01", "ON", "ACCESS GRANTED"), ("user_02", "ON", "ACCESS DENIED"),
    ("user_03", "GO", "ACCESS DENIED"), ("user_03", "STOP", "ACCESS GRANTED")])
def test_authorized_routing_only_after_verification(monkeypatch, user, word, expected):
    calls, system = routing_doubles(monkeypatch, word=word)
    result = pipeline.run_pipeline("face", "voice", "finger", user, system)
    assert result["result"] == expected
    assert result["authenticated"]
    assert calls.count("extract:voice") == 1
    assert calls.index("fusion3") < calls.index("command")
    assert (result["action"] is not None) == (expected == "ACCESS GRANTED")


@pytest.mark.parametrize("failed", pipeline.STAGES)
def test_every_failed_gate_blocks_execution(monkeypatch, failed):
    calls, system = routing_doubles(monkeypatch, fail_stage=failed)
    state = {"enabled": False, "running": False}
    result = pipeline.run_pipeline("face", "voice", "finger", "user_01", system, state)
    assert result["result"] == "VERIFICATION FAILED"
    assert result["action"] is None and "command" not in calls
    assert state == {"enabled": False, "running": False}
    failed_index = list(pipeline.STAGES).index(failed)
    assert all(row["status"] == "NOT RUN" for row in result["trace"][failed_index + 1:])


def test_unknown_command_never_executes(monkeypatch):
    calls, system = routing_doubles(monkeypatch, word="UNKNOWN")
    result = pipeline.run_pipeline("face", "voice", "finger", "user_01", system)
    assert result["result"] == "UNKNOWN COMMAND" and result["action"] is None
    assert result["permission"] == "NOT RUN"


def test_missing_models_fails_closed(monkeypatch):
    def missing():
        raise FileNotFoundError("Deliberately missing for control-flow test.")

    monkeypatch.setattr(pipeline, "load_system", missing)
    result = pipeline.run_pipeline("absent", "absent", "absent", "user_01")
    assert result["result"] == "VERIFICATION FAILED"
    assert result["action"] is None


@pytest.fixture(scope="module")
def sample_manifest():
    return pd.read_csv(MANIFESTS / "samples.csv", keep_default_na=False)


@pytest.fixture(scope="module")
def real_data(sample_manifest):
    if not any((PROCESSED / modality).is_dir() for modality in ("face", "voice", "fingerprint")):
        pytest.skip("Full experiment samples are absent from the demo checkout.")
    return sample_manifest


@pytest.fixture(scope="module")
def real_system():
    return load_system()


def test_no_file_or_decoded_content_overlap(real_data):
    assert validate_manifest(real_data)
    for modality, group in real_data.groupby("modality"):
        assert not group.sha256.duplicated().any(), modality
        assert not group.decoded_sha256.duplicated().any(), modality


@pytest.mark.parametrize("modality", ["voice", "fingerprint"])
def test_public_source_identity_is_consistent(sample_manifest, modality):
    group = sample_manifest[sample_manifest.modality == modality]
    assert group.groupby("user_id").source_identity.nunique().tolist() == [1, 1, 1]
    assert group.source_identity.nunique() == 3


def test_exact_public_splits(sample_manifest):
    voice = sample_manifest[sample_manifest.modality == "voice"]
    counts = voice.groupby(["user_id", "command", "split"]).size()
    for (user, command, split), count in counts.items():
        assert count == {"train": 3, "validation": 1, "test": 1}[split]
    fingers = sample_manifest[sample_manifest.modality == "fingerprint"]
    for (user, split), count in fingers.groupby(["user_id", "split"]).size().items():
        assert count == {"train": 5, "validation": 1, "test": 2}[split]


def test_live_test_has_three_real_held_out_folders(real_data):
    live = pd.read_csv(LIVE / "manifest.csv")
    assert len(live) == 3
    assert set(live.display_name) == set(real_data.display_name)
    for row in live.to_dict("records"):
        folder = LIVE / row["display_name"]
        assert {path.name for path in folder.iterdir()} == {"face.jpg", "voice.wav", "fingerprint.png"}
        identifiers = json.loads(row["sample_ids"])
        samples = real_data[real_data.sample_id.isin(identifiers)]
        assert len(samples) == 3 and samples.split.eq("test").all()
        paths = [ROOT / row[name] for name in ("face_path", "voice_path", "fingerprint_path")]
        assert [file_hash(path) for path in paths] == json.loads(row["file_hashes"])
        assert face_features(paths[0]).shape == (8878,)
        assert voice_features(paths[1]).shape == (30,)
        assert fingerprint_features(paths[2]).shape == (8110,)


def test_all_twelve_models_use_expected_classifiers(real_system):
    assert len(real_system["models"]) == 6
    assert sum(len(values) for values in real_system["models"].values()) == 12
    for variants in real_system["models"].values():
        assert variants["knn"]["classifier"].n_neighbors == 3
        assert variants["svm"]["classifier"].kernel == "linear"


def test_preprocessing_and_models_were_fit_on_training_only(real_system, sample_manifest):
    training = sample_manifest[sample_manifest.split == "train"]
    training_ids = set(training.sample_id)
    for modality, preprocessor in real_system["preprocessors"].items():
        expected = len(training[training.modality == modality])
        assert preprocessor[0].n_samples_seen_ == expected
        assert preprocessor[1].n_samples_ == expected
        for artifact in real_system["models"][modality].values():
            assert set(artifact["training_ids"]) <= training_ids


def test_fusion_models_use_concatenated_features(real_system):
    dimensions = real_system["index"]["dimensions"]
    face_size, voice_size, finger_size = [dimensions[name]["after_pca"] for name in ("face", "voice", "fingerprint")]
    assert real_system["models"]["face_voice"]["knn"]["n_features"] == face_size + voice_size
    assert real_system["models"]["fusion3"]["svm"]["n_features"] == face_size + voice_size + finger_size
    assert concatenate_features(np.array([1, 2]), np.array([3])).tolist() == [1, 2, 3]


def test_fusion_manifests_have_no_repeated_components(sample_manifest):
    lookup = sample_manifest.set_index("sample_id")
    for name in ("face_voice_pairs.csv", "three_modality_triplets.csv"):
        manifest = pd.read_csv(MANIFESTS / name)
        for column in ("face_id", "voice_id", "fingerprint_id"):
            if column not in manifest:
                continue
            assert not manifest[column].duplicated().any()
            for record in manifest.to_dict("records"):
                source = lookup.loc[record[column]]
                assert source.user_id == record["user_id"] and source.split == record["split"]


def test_actual_wrong_claim_runs_real_rejection_logic(real_system, real_data):
    samples = real_data[(real_data.modality == "face") & (real_data.split == "test")]
    results = [verify_face(ROOT / row.path, claim, real_system)
               for row in samples.itertuples() for claim in real_system["identities"] if claim != row.user_id]
    assert len(results) == 48
    assert all(row["distance"] is not None for row in results)
    assert any(not row["accepted"] for row in results)
    assert all(row["accepted"] == (row["predicted_identity"] == row["claimed_identity"] and
                                    row["distance"] <= row["threshold"]) for row in results)


def test_corrupt_missing_and_nonfinite_data_fail_safely(real_system):
    result = pipeline.run_pipeline(b"not an image", b"not audio", b"not a fingerprint", "user_01", real_system)
    assert result["result"] == "VERIFICATION FAILED" and result["action"] is None
    assert result["trace"][0]["status"] == "REJECTED"
    size = real_system["models"]["face"]["svm"]["n_features"]
    decision = verify_vector("face", np.full(size, np.nan), "user_01", real_system)
    assert not decision["accepted"]


def test_verification_metric_numerators_and_denominators():
    from evaluate import verification_metrics

    measured = verification_metrics([True, True, False, False], [True, False, True, False])
    assert measured["false_accepts"] == 1 and measured["impostor_attempts"] == 2
    assert measured["false_rejects"] == 1 and measured["genuine_attempts"] == 2
    assert measured["far"] == 0.5 and measured["frr"] == 0.5
    assert measured["verification_precision"] == 0.5
    assert measured["verification_recall"] == 0.5
    assert measured["verification_f1"] == 0.5


def test_wrong_permitted_command_is_not_a_correct_action():
    from evaluate import action_metrics

    arithmetic_fixture = pd.DataFrame([{"source_command": "OFF", "expected_command": "OFF",
                                        "executed_command": "ON", "expected_action": True,
                                        "action_executed": True, "permission_violation": False}])
    measured = action_metrics(arithmetic_fixture)
    assert measured["action_execution_decision_accuracy"] == 1.0
    assert measured["exact_action_accuracy"] == 0.0
    assert measured["unexpected_actions_vs_source_word"] == 1
    assert measured["wrong_executed_commands"] == 1


def test_validation_threshold_reports_match_actual_runtime_rules(real_system, real_data):
    from commands import recognize_command
    from evaluate import verification_metrics
    from train import task_data
    from verification import extract_vector

    vectors = {row.sample_id: extract_vector(row.modality, ROOT / row.path, real_system)
               for row in real_data.itertuples()}
    for task, splits in task_data(real_data, vectors).items():
        values, labels, _ = splits["validation"]
        for algorithm, artifact in real_system["models"][task].items():
            if task == "command":
                genuine = np.isin(labels, ["ON", "OFF", "GO", "STOP"])
                accepted = [recognize_command(value, real_system, algorithm)["supported"] for value in values]
            else:
                genuine, accepted = [], []
                for value, actual in zip(values, labels):
                    for claim in real_system["identities"]:
                        genuine.append(actual == claim)
                        accepted.append(verify_vector(task, value, claim, real_system, algorithm)["accepted"])
            measured = verification_metrics(genuine, accepted)
            assert measured["far"] == pytest.approx(artifact["validation_far"]), (task, algorithm)
            assert measured["frr"] == pytest.approx(artifact["validation_frr"]), (task, algorithm)


def test_notebooks_json_and_code_are_valid():
    import ast
    import nbformat

    for path in (ROOT / "01_Biometric_System.ipynb", ROOT / "02_Streamlit_Demo.ipynb"):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert all(cell.metadata.get("language") in ("python", "markdown") for cell in notebook.cells)
        assert all(cell.metadata.get("id") for cell in notebook.cells)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert code_cells
        for cell in code_cells:
            ast.parse(cell.source)


def test_camera_identification_rejects_when_distance_exceeds_threshold(real_system):
    from camera_face import load_camera_system
    from camera_identification import identify_camera_face

    photo = (ROOT / "datasets/demo/Abdelrahman/face.jpg").read_bytes()
    camera = load_camera_system(real_system["identities"])
    assert identify_camera_face(photo, camera) == ("user_01", None)

    selected = camera["selected"]["face"]
    models = camera["models"]["face"]
    restricted = {**models[selected], "threshold": 0.0}
    stricter_camera = {**camera, "models": {"face": {**models, selected: restricted}}}
    identity, error = identify_camera_face(photo, stricter_camera)
    assert identity is None
    assert error.startswith("Face not recognized")


@pytest.mark.parametrize("folder, name, other, other_name", [
    ("Abdelrahman", "Abdelrahman", "kareem", "Kareem"),
    ("kareem", "Kareem", "renad", "Renad"),
    ("renad", "Renad", "Abdelrahman", "Abdelrahman"),
])
def test_streamlit_staged_sign_in(folder, name, other, other_name):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    demo = ROOT / "datasets" / "demo"
    assert len(app.camera_input) == 1
    assert not app.selectbox and not app.file_uploader
    assert "Model Evaluation" not in [heading.value for heading in app.subheader]

    empty_frame = io.BytesIO()
    Image.new("RGB", (320, 240), "gray").save(empty_frame, format="JPEG")
    app.camera_input[0].upload("empty.jpg", empty_frame.getvalue()).run()
    assert app.error[0].value.startswith("No face detected")
    assert app.camera_input and not app.file_uploader

    app.camera_input[0].upload("face.jpg", (demo / folder / "face.jpg").read_bytes()).run()
    assert [item.value for item in app.success] == [f"Identified as {name}."]
    assert not app.camera_input and not app.selectbox
    assert [widget.label for widget in app.file_uploader] == ["Voice recording"]
    assert [button.label for button in app.button] == ["Start over"]

    app.file_uploader[0].set_value(("voice.wav", (demo / other / "voice.wav").read_bytes(), "audio/wav")).run()
    assert app.error[0].value.startswith("Voice did not verify")
    assert [widget.label for widget in app.file_uploader] == ["Voice recording"]

    voice = ("voice.wav", (demo / folder / "voice.wav").read_bytes(), "audio/wav")
    app.file_uploader[0].set_value(voice).run()
    assert [item.value for item in app.success] == [f"Voice verified for {name}."]
    assert not app.camera_input and not app.selectbox
    assert [widget.label for widget in app.file_uploader] == ["Fingerprint image"]
    assert {button.label for button in app.button} == {"Change recording", "Start over"}

    app.file_uploader[0].set_value(("fingerprint.png", (demo / other / "fingerprint.png").read_bytes(), "image/png")).run()
    assert app.error[0].value.startswith("Fingerprint did not verify")
    assert [widget.label for widget in app.file_uploader] == ["Fingerprint image"]

    next(button for button in app.button if button.label == "Change recording").click().run()
    assert [item.value for item in app.success] == [f"Identified as {name}."]
    assert [widget.label for widget in app.file_uploader] == ["Voice recording"]
    assert not app.file_uploader[0].value and not app.error
    app.file_uploader[0].set_value(voice).run()
    assert [widget.label for widget in app.file_uploader] == ["Fingerprint image"]
    assert not app.file_uploader[0].value and not app.error

    app.file_uploader[0].set_value(("fingerprint.png", (demo / folder / "fingerprint.png").read_bytes(), "image/png")).run()
    assert [item.value for item in app.success] == [f"Signed in as {name}."]
    assert not app.camera_input and not app.file_uploader and not app.selectbox
    assert [button.label for button in app.button] == ["Start over"]

    app.button[0].click().run()
    assert not app.camera_input[0].value
    assert not app.file_uploader and not app.success and not app.error
    app.camera_input[0].upload("face.jpg", (demo / other / "face.jpg").read_bytes()).run()
    assert [item.value for item in app.success] == [f"Identified as {other_name}."]
    assert [widget.label for widget in app.file_uploader] == ["Voice recording"]
    assert not app.file_uploader[0].value
    assert not app.exception
