"""Classical face verification for camera photos, trained on the existing portraits."""

import io
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage import data
from skimage.feature import Cascade
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import MANIFESTS, MODELS, ROOT, SEED
from dataset_setup import file_hash, validate_manifest
from features import hog_lbp
from verification import calibrate_threshold, transform_features, verify_vector

CAMERA_MODEL = MODELS / "camera_face.joblib"
ORIGINAL = (1.0, 0.0, 0.0, False, 1.0)
# Crop zoom, horizontal/vertical offset, mirror and gamma. Only training uses variants.
TRAIN_VARIANTS = (
    ORIGINAL,
    (1.0, 0.0, 0.0, True, 1.0),
    (0.9, -0.035, 0.025, False, 0.8),
    (1.1, 0.035, -0.025, False, 1.25),
    (0.9, 0.035, -0.025, True, 1.25),
    (1.1, -0.035, 0.025, True, 0.8),
)


def open_face_image(source):
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    with Image.open(source) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def locate_face(image, detector):
    """Find the dominant frontal face and return its center and width in source pixels."""
    preview = ImageOps.autocontrast(image.convert("L"), cutoff=1)
    preview.thumbnail((320, 320), Image.Resampling.LANCZOS)
    shortest_side = min(preview.size)
    minimum = max(24, int(0.28 * shortest_side))
    if shortest_side < minimum:
        return None
    boxes = detector.detect_multi_scale(
        img=np.asarray(preview), scale_factor=1.15, step_ratio=1,
        min_size=(minimum, minimum), max_size=(shortest_side, shortest_side),
        min_neighbor_number=5,
    )
    if not boxes:
        return None
    face = max(boxes, key=lambda box: box["width"])
    ratio = image.width / preview.width
    return ((face["c"] + face["width"] / 2) * ratio,
            (face["r"] + face["height"] / 2) * ratio,
            face["width"] * ratio)


def camera_features(image, box, variant=ORIGINAL):
    center_x, center_y, face_width = box
    zoom, shift_x, shift_y, mirror, gamma = variant
    side = min(face_width * 1.15 * zoom, min(image.size))
    left = max(0, min(image.width - side, center_x + face_width * shift_x - side / 2))
    top = max(0, min(image.height - side, center_y + face_width * shift_y - side / 2))
    crop = image.resize((128, 128), Image.Resampling.LANCZOS,
                        box=(left, top, left + side, top + side)).convert("L")
    crop = ImageOps.autocontrast(crop, cutoff=1)
    if mirror:
        crop = ImageOps.mirror(crop)
    if gamma != 1:
        crop = crop.point([round(255 * (level / 255) ** gamma) for level in range(256)])
    return hog_lbp(np.asarray(crop))


def source_hashes():
    sources = (Path(__file__).resolve(), ROOT / "features.py", ROOT / "verification.py",
               MANIFESTS / "samples.csv", MANIFESTS / "identities.json")
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in sources}


def load_camera_system(identities):
    camera = joblib.load(CAMERA_MODEL)
    if camera["source_hashes"] != source_hashes():
        raise ValueError("Camera model inputs changed. Run python camera_face.py to retrain.")
    return {**camera, "identities": identities}


def verify_camera_face(source, claim, camera):
    try:
        image = open_face_image(source)
        detector = Cascade(data.lbp_frontal_face_cascade_filename())
        box = locate_face(image, detector)
        if box is None:
            raise ValueError("No face detected. Clear the photo and center your face in the camera.")
        vector = transform_features(camera["preprocessors"]["face"], camera_features(image, box))
        return verify_vector("face", vector, claim, camera)
    except OSError:
        return {"accepted": False, "predicted_identity": None,
                "explanation": "Could not read the photo. Clear it and take another."}
    except ValueError as error:
        return {"accepted": False, "predicted_identity": None, "explanation": str(error)}


def portrait_features(samples, detector, variants):
    features, labels = [], []
    for sample in samples.itertuples():
        image = open_face_image(ROOT / sample.path)
        box = locate_face(image, detector)
        if box is None:
            # The enrolled portraits are pre-framed. Live captures must have a detected face.
            box = (image.width / 2, image.height / 2, min(image.size) / 1.15)
        for variant in variants:
            features.append(camera_features(image, box, variant))
            labels.append(sample.user_id)
    return np.asarray(features, dtype=np.float32), np.asarray(labels)


def train_camera_models():
    manifest = pd.read_csv(MANIFESTS / "samples.csv", keep_default_na=False)
    samples = manifest[manifest.modality.eq("face") & manifest.split.isin(["train", "validation"])]
    validate_manifest(samples)
    detector = Cascade(data.lbp_frontal_face_cascade_filename())
    training = samples[samples.split.eq("train")]
    validation = samples[samples.split.eq("validation")]
    train_raw, train_labels = portrait_features(training, detector, TRAIN_VARIANTS)
    val_raw, val_labels = portrait_features(validation, detector, (ORIGINAL,))

    preprocessor = make_pipeline(StandardScaler(), PCA(n_components=0.95, svd_solver="full",
                                                       whiten=True, random_state=SEED))
    preprocessor.fit(train_raw)
    train_vectors = np.asarray([transform_features(preprocessor, row) for row in train_raw])
    val_vectors = np.asarray([transform_features(preprocessor, row) for row in val_raw])
    models, ranking = {}, {}
    classifiers = {"knn": KNeighborsClassifier(n_neighbors=3), "svm": SVC(kernel="linear")}
    for name, classifier in classifiers.items():
        classifier.fit(train_vectors, train_labels)
        predicted = classifier.predict(val_vectors)
        macro_f1 = f1_score(val_labels, predicted, average="macro")
        accuracy = accuracy_score(val_labels, predicted)
        ranking[name] = (macro_f1, accuracy, name == "knn")
        calibration = calibrate_threshold(classifier, train_vectors, train_labels, val_vectors, val_labels)
        calibration.pop("validation_trials")
        models[name] = {**calibration, "classifier": classifier, "n_features": train_vectors.shape[1],
                        "validation_macro_f1": macro_f1, "validation_accuracy": accuracy}
        print(f"{name}: validation macro F1 {macro_f1:.3f}, FAR {calibration['validation_far']:.3f}, "
              f"FRR {calibration['validation_frr']:.3f}", flush=True)

    selected = max(ranking, key=ranking.get)
    artifact = {"source_hashes": source_hashes(), "preprocessors": {"face": preprocessor},
                "selected": {"face": selected}, "models": {"face": models},
                "training_ids": training.sample_id.tolist(), "validation_ids": validation.sample_id.tolist(),
                "training_variants_per_portrait": len(TRAIN_VARIANTS)}
    MODELS.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=MODELS, suffix=".joblib", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        joblib.dump(artifact, temporary_path)
        temporary_path.replace(CAMERA_MODEL)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Camera face model: {selected.upper()} ({len(training)} training portraits, "
          f"{len(train_vectors)} variants, {len(validation)} validation portraits).")


if __name__ == "__main__":
    train_camera_models()
