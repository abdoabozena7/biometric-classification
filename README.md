# Traditional ML Biometric Verification

Run the sign-in app below or open **[02_Streamlit_Demo.ipynb](02_Streamlit_Demo.ipynb)**
to exercise it with the included samples. **[01_Biometric_System.ipynb](01_Biometric_System.ipynb)**
walks through the feature pipelines, saved predictions, and measured results.
Both notebooks share the Python helpers used by Streamlit.

**This is an educational virtual-identity demonstration, not production security.**
The face, public voice and public fingerprint under one user ID belong to unrelated
physical people. Real multimodal authentication would require collecting all three
modalities from the same person, with consent and separate capture sessions.

## Run

Use Python 3.13. The checkout includes the saved models, preprocessors, model
manifests, and three sets of demo samples. After installing the dependencies,
you can start the app and run both notebooks directly from the checkout.

From a Windows PowerShell terminal in this project folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

On macOS or Linux:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Local application: **http://127.0.0.1:8501**. If occupied, choose another port.
In VS Code, select the Python interpreter from `.venv` as the notebook kernel.
To verify the checkout, run `python -m pytest -q` in the activated environment.

To reproduce training and held-out evaluation, prepare your authorized full
datasets using [the dataset source instructions](DATASET_SOURCES.md). With the
original files and dataset cache available locally, run
`python dataset_setup.py --local-only`, `python train.py`, `python camera_face.py`,
and `python evaluate.py`.
`--faces-only` is a partial-data diagnostic mode; it explicitly marks other
modalities BLOCKED and is not full project completion.

## Sign In

1. Allow camera access, center your face, and take a photo. The saved camera
   model identifies the enrolled person when the match passes its distance check.
2. Upload `voice.wav` from that person's folder in `datasets/demo`.
3. After the voice check passes, upload `fingerprint.png` from the same folder.

The app shows one step at a time and checks both uploads against the person
identified by the camera. You can retry a rejected input or start over with a
new photo. The demo folders provide reference photos and sample uploads for the
three virtual accounts.
Abdelrahman's demo fingerprint comes from validation. The other demo files
come from held-out test samples.

For the camera stage, `camera_face.py` locates the face with an LBP cascade,
crops it, and normalizes grayscale contrast before extracting HOG and LBP.
Small crop shifts, mirroring and gamma changes create six variants per training
portrait. Scaling and PCA fit those training variants. The original validation
portraits select between KNN(3) and linear SVM and calibrate the acceptance distance.
The original face and fusion experiments continue to use their saved models.

With the published files, the main notebook extracts features from the demo
samples, runs saved-model predictions, and displays the recorded evaluation.
With the full prepared dataset available locally, it also runs the training
example and recomputes the held-out metrics. `REBUILD_DATA` and `RETRAIN` default
to `False`.

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

The main notebook runs the command and permissions pipeline with the included
demo triplets. With local `live_test` files, it uses the original held-out
triplets for that exercise. The sign-in app uses the three individual biometric
checks.

## Tests and Bonuses

The test suite checks saved-model inference and the Streamlit flow with the
included samples. Tests that read the full prepared dataset or original
held-out triplets require those files locally. Both notebooks can run with the
published demo files.

| Bonus | Honest Status |
| --- | --- |
| Verification-based speech commands | Implemented and exercised through all gates and permissions; command accuracy is poor |
| Real-time verification | Camera path implemented; genuine camera capture/inference demonstration NOT VERIFIED |
| Custom dataset | Supplied custom face part satisfied; team-collected custom voice part NOT SATISFIED |
| Fingerprint verification | Real DB1_B classifiers and feature fusion implemented and evaluated; high FRR remains |

For simple English and Egyptian Arabic explanations of every module and concept,
read [ORAL_DEFENSE.md](ORAL_DEFENSE.md).

Before sharing this project, confirm consent for the supplied photos and the
redistribution rights for the biometric datasets and derived artifacts. See
[DATASET_SOURCES.md](DATASET_SOURCES.md) for source terms and attribution. Load
Joblib files only from a trusted copy of the project.
