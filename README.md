# Traditional ML Biometric Verification

Start with **[01_Biometric_System.ipynb](01_Biometric_System.ipynb)**.
It is the main teaching notebook: short sequential cells, separate pipelines,
actual feature-function source, real samples, training examples and saved outputs.
Then open **[02_Streamlit_Demo.ipynb](02_Streamlit_Demo.ipynb)** for the local UI.
There are two notebooks, not a complicated notebook series or a custom framework.
The small Python helpers are shared by both notebooks and Streamlit.

**This is an educational virtual-identity demonstration, not production security.**
The face, public voice and public fingerprint under one user ID belong to unrelated
physical people. Real multimodal authentication would require collecting all three
modalities from the same person, with consent and separate capture sessions.

## Run

Before running dataset setup from a fresh clone, place your authorized face
samples under `datasets/images/` in three subject folders. See
[Dataset Sources and Provenance](DATASET_SOURCES.md) for the input layout and
sources used by the setup script. The repository includes source code, notebooks,
and aggregate results. Prepare the datasets and train the models locally.

In VS Code, select `.venv/Scripts/python.exe` as the notebook kernel after
creating the environment. The published notebook omits the embedded
biometric sample images and recording.

From a Windows PowerShell terminal in this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe dataset_setup.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Do not recreate the environment if using the prepared one. With an already
activated environment, `pip install -r requirements.txt`, `python dataset_setup.py`,
`python train.py`, `python evaluate.py`, and `streamlit run app.py` also work.

Local application: **http://127.0.0.1:8501**. If occupied, choose another port.
Cached offline data preparation: `python dataset_setup.py --local-only`.
`--faces-only` is a partial-data diagnostic mode; it explicitly marks other
modalities BLOCKED and is not full project completion.

## Sign In

1. Choose an enrolled account. Allow camera access in your browser, center your
   face, and take a photo.
2. Once the face passes, upload `voice.wav` from your account folder in
   `datasets/demo`.
3. Once the voice passes, upload `fingerprint.png` from the same folder.

Each input must pass its saved model's claimed-identity check. A rejected step
can be retried, and changing the account, photo or recording clears later steps.
The three demo folders include reference photos and sample uploads for the
virtual accounts. The app takes the face image from the camera.
Abdelrahman's demo fingerprint comes from validation. The other demo files
come from held-out test samples.

The main notebook defaults to `REBUILD_DATA=False` and `RETRAIN=False`. Run All
does not silently replace splits or retrain saved models. Its small demonstration
loop fits temporary KNN/SVM classifiers on training data only, then prints their
validation predictions; the app always uses the saved selected artifacts.

## The Pipelines

```text
Image -> center-fit 128x128 -> HOG + LBP (+ RGB hist for face)
      -> training-only StandardScaler -> training-only PCA -> identity model

WAV -> mono 16kHz -> MFCC + ZCR + RMS means/stds -> 30 features
    -> training-only StandardScaler/PCA -> speaker model OR command model

[transformed face | transformed voice] -> dedicated fusion identity model
[transformed face | transformed voice | transformed fingerprint]
    -> dedicated three-modality identity model

Claim -> face -> speaker -> face+voice -> fingerprint -> fusion3
      -> SAME voice vector -> command -> permissions -> simulated action
      Any rejection: STOP; all following stages: NOT RUN
```

Every verifier requires both `predicted_identity == claimed_identity` and distance
to the claimed user's training centroid within a validation-calibrated threshold.
The threshold minimizes validation FAR + FRR, then lower FAR and lower threshold.
Command calibration uses its actual supported/UNKNOWN decision rule, not
claimed-identity trials. Training preparation and inference use the same
per-sample float32 transform path so boundary decisions agree numerically.
It is not automatic acceptance of the closest enrolled person.

## Real Data

| Virtual User | Supplied Face Folder | Role | Original Speaker | Original Finger |
| --- | --- | --- | --- | --- |
| user_01 | Abdelrahman | Admin | 834f03fe | FVC2004 DB1_B:101 |
| user_02 | kareem | Employee | 8e05039f | FVC2004 DB1_B:102 |
| user_03 | renad | Guest | 3bfd30e6 | FVC2004 DB1_B:103 |

Names are discovered automatically, not hardcoded into dataset processing or UI.

| Dataset/Pairing | Total | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| Faces: 40/person | 120 | 72 | 24 | 24 |
| Voices: 6 words x 5 utterances/person | 90 | 54 | 18 | 18 |
| Fingerprints: 8 impressions/person | 24 | 15 | 3 | 6 |
| Face + voice pairs, no source reuse within this manifest | 90 | 54 | 18 | 18 |
| Three-modality triplets, no source reuse within this manifest | 24 | 15 | 3 | 6 |

Per speaker/word: 3/1/1 utterances. Words: ON, OFF, GO, STOP plus YES/NO as UNKNOWN.
The speaker model also uses YES/NO, since speaker identity is separate from word.
The voice experiment uses custom utterance-level splits, not official benchmark
results. The selected speakers each have one consistent source identity.

File bytes and decoded content are checked for leakage. An initially considered
speaker, b5cf6ea8, had duplicate recordings and was rejected before training.
There are no corrupt or identical supplied face files, but 54 similar-image pairs
for Abdelrahman and 2 for kareem were flagged. These are warnings, not proof of
duplicate captures; capture-session independence cannot be established from files.
Training/test grouping by a true capture session was impossible without session
metadata. Thus apparently high face accuracy may not generalize to new photos.

Sources, licensing cautions and manual recovery: [DATASET_SOURCES.md](DATASET_SOURCES.md).

## Twelve Saved Classifiers

Tasks: `face`, `voice`, `fingerprint`, `face_voice`, `fusion3`, `command`.
Each has `{task}_knn.joblib` and `{task}_svm.joblib`.
Only `KNeighborsClassifier(n_neighbors=3)` and `SVC(kernel="linear")` are used.
The saved preprocessing has raw/PCA dimensions 8878/57 for face, 30/16 for voice,
and 8110/13 for fingerprint; concatenated fusion dimensions are 73 and 86.
PCA keeps approximately 95% training variance, whitens retained components, and
each block is divided by sqrt(its dimension) for balanced concatenation.

Selection uses validation macro-F1, then validation accuracy, then KNN on exact
ties. SVM won the five identity/fusion tasks; KNN won command recognition.

## Actual Results

Selected models on held-out data, not illustrative scores:

| Task | Model | Classification Accuracy | Macro F1 | Verification FAR | Verification FRR |
| --- | --- | ---: | ---: | ---: | ---: |
| Face | SVM | 100.00% | 1.0000 | 0/48 | 1/24 |
| Voice | SVM | 100.00% | 1.0000 | 0/36 | 0/18 |
| Fingerprint | SVM | 83.33% | 0.8222 | 0/12 | 3/6 |
| Face + voice | SVM | 100.00% | 1.0000 | 0/36 | 0/18 |
| Three modalities | SVM | 100.00% | 1.0000 | 0/12 | 3/6 |
| Command | KNN | 55.56% | 0.5233 | 2/6 unsupported accepted | 0/12 supported rejected |

Classification accuracy is not verification acceptance accuracy. A correctly
identified sample can still fail distance verification. The command rate is
unsupported-word rejection, not biometric FAR. Six test triplets generate 6
genuine and 12 wrong-claim attempts, not 18 independent source observations.

Selected hierarchy: 3/6 genuine authentications, 0/12 impostor authentications,
83.33% claim-decision accuracy. It executed two simulated actions; both were
inconsistent with the source word because GO was predicted STOP. There were zero
permission-dictionary violations for the recognized word. This distinction is
essential: correct authorization cannot repair a wrongly recognized command.
Do not use this recognizer to operate real equipment.

Exact intended-action accuracy is 66.67%; the separate Boolean execute/deny
accuracy is 72.22%. Correctly deciding to execute something does not establish
that the correct command was executed. The initial run is preserved privately
in results/initial_run.zip. A source review corrected command calibration and
exact-action scoring using validation data only. The rerun reuses the already-seen
test set as a regression check, not a new blind experiment or claimed improvement.

Both algorithms' full metrics, rate denominators and confusion matrices are in
[results/metrics/test_metrics.csv](results/metrics/test_metrics.csv) and
[results/metrics/hierarchy_metrics.csv](results/metrics/hierarchy_metrics.csv).
The complete measured account is [FINAL_REPORT.md](FINAL_REPORT.md).

## Results for the Original Prepared Test Triplets

- Abdelrahman: face, speaker and face+voice pass; fingerprint rejects; STOP.
- kareem: all five biometric gates pass; OFF is predicted ON; Employee denied.
- renad: face, speaker and face+voice pass; fingerprint rejects; STOP.

These fixed held-out files were chosen before evaluation. No threshold or demo
file was replaced to create a successful-looking presentation. The JPEG demo
face is a re-encoding of a test source, not a separate independent observation.

The original triplets and full command and permissions pipeline are available
in the notebooks for evaluation. The sign-in page uses the three individual
biometric checks and the samples under `datasets/demo`.

## Tests and Bonuses

31 automated tests passed with no skips. Doubles are limited to routing tests;
real integration tests check files, held-out inference, all twelve classifiers,
concatenation, preprocessing sample counts and actual Streamlit interactions.
Both notebooks executed cleanly: 16 and 5 code cells. Desktop/mobile browser
screenshots were inspected, images/audio loaded, an impostor was rejected, and
the no-camera state disabled verification. Browser camera permission prevented
a genuine new capture; real webcam inference remains **UNVERIFIED**.

| Bonus | Honest Status |
| --- | --- |
| Verification-based speech commands | Implemented and exercised through all gates and permissions; command accuracy is poor |
| Real-time verification | Camera path implemented; genuine camera capture/inference demonstration NOT VERIFIED |
| Custom dataset | Supplied custom face part satisfied; team-collected custom voice part NOT SATISFIED |
| Fingerprint verification | Real DB1_B classifiers and feature fusion implemented and evaluated; high FRR remains |

For simple English and Egyptian Arabic explanations of every module and concept,
read [ORAL_DEFENSE.md](ORAL_DEFENSE.md).

Biometric files, models and prepared data are excluded from Git by default.
Do not publish supplied photos, executed notebooks containing them, public voice files, fingerprints or model
representations without the relevant consent and dataset rights. Joblib files
must be loaded only from this trusted local training run, never an unknown source.
