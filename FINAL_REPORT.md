# Actual Project Results

## Delivery Status

The offline educational system is implemented, trained and exercised on real data.
Both notebooks ran from clean kernels. This is **not a claim of production-safe
authentication, accurate command recognition, or completed real-webcam bonus**.
Camera permission prevented a genuine capture in the integrated browser; no saved
face or synthetic camera frame was used as a live demonstration. Team-collected
voice remains NOT SATISFIED by design because public recordings were requested.

## Actual Directory Structure

```text
01_Biometric_System.ipynb       main explained workflow, 16 executed code cells
02_Streamlit_Demo.ipynb         app explanation/checks/launcher, 5 executed code cells
datasets/images/
  Abdelrahman/                 40 supplied originals, preserved
  kareem/                      40 supplied originals, preserved
  renad/                       40 supplied originals, preserved
data/
  raw/                         real source cache, speech metadata, official DB1_B.zip
  processed/                   separate modality/train/validation/test paths
  manifests/                   samples, identities, pairs, triplets, acquisition/quality audits
models/                        twelve classifiers, preprocessors, model_index.json
live_test/
  Abdelrahman/                 face.jpg, voice.wav, fingerprint.png
  kareem/                      face.jpg, voice.wav, fingerprint.png
  renad/                       face.jpg, voice.wav, fingerprint.png
  manifest.csv                 all source IDs, split and hashes
results/
  metrics/                     comparison, test metrics, thresholds, attempt traces, summaries
  confusion_matrices/          actual classification/verification/action PNG and CSV matrices
  initial_run.zip              private archive of pre-review models, results and affected source
config.py
dataset_setup.py
features.py
train.py
verification.py
fusion.py
commands.py
permissions.py
pipeline.py
evaluate.py
app.py
tests/test_system.py
pytest.ini
.streamlit/config.toml
requirements.txt
README.md
DATASET_SOURCES.md
ORAL_DEFENSE.md
FINAL_REPORT.md
PROJECT_STATUS.md
PRODUCT.md
.gitignore
```

The existing datasets/images location replaces the illustrative input_faces
folder; originals were not moved or duplicated into a second source dataset.
The local .venv is a development environment, not part of the project source.

## Data Sources and Virtual Identities

- Speech Commands v0.02: https://www.tensorflow.org/datasets/catalog/speech_commands
- Download distribution: https://huggingface.co/datasets/pollen-robotics/speech-commands-v0.02
- Real fingerprint subset: http://bias.csr.unibo.it/fvc2004/Downloads/DB1_B.zip
- Official FVC source page: https://bias.csr.unibo.it/fvc2004/download.asp

| Virtual ID | Face/Display Name | Role | Original Speaker | Original Finger |
| --- | --- | --- | --- | --- |
| user_01 | Abdelrahman | Admin | 834f03fe | FVC2004_DB1_B:101 |
| user_02 | kareem | Employee | 8e05039f | FVC2004_DB1_B:102 |
| user_03 | renad | Guest | 3bfd30e6 | FVC2004_DB1_B:103 |

These are educational associations of unrelated physical subjects. Speech is
CC BY 4.0 as declared by its distribution; preserve attribution. FVC publicly
provides the official B sample archive, but unrestricted redistribution permission
was not established. Do not publish biometric files or claim an open-data license
for FVC. HTTPS hostname validation failed, so the provider's advertised HTTP path
was used without disabling TLS verification. See DATASET_SOURCES.md for details.

## Counts, Splits and Integrity

| Modality/Construction | Unique Selected Sources/Observations | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| Face | 120 source files | 72 | 24 | 24 |
| Voice | 90 distinct decoded utterances | 54 | 18 | 18 |
| Fingerprint | 24 distinct source impressions | 15 | 3 | 6 |
| Face+voice pairs | 90 no-reuse pairs from those sources | 54 | 18 | 18 |
| Three-modality triplets | 24 no-reuse triplets from those sources | 15 | 3 | 6 |

Per person: face 24/8/8; each of six recorded words 3/1/1; finger 5/1/2.
Supported words are exactly ON/OFF/GO/STOP; genuine YES/NO provide UNKNOWN data.
No exact hash or decoded-content overlap crosses splits. No fingerprint is
multiplied into many independent triplets. Pair and triplet manifests share
some underlying sources; they are different task constructions, not new captures.

40 face files/person were available, not 20. All decode, with no exact duplicates.
There are 54 near-similar pairs for Abdelrahman and 2 for kareem. No capture-session
metadata exists; unique files do not establish independent acquisitions. The
reported face accuracy is limited by this possible related-image dependence.
An initial speaker candidate was rejected for duplicated audio before training.

## Saved Models and Dimensions

```text
face_knn.joblib              face_svm.joblib
voice_knn.joblib             voice_svm.joblib
fingerprint_knn.joblib       fingerprint_svm.joblib
face_voice_knn.joblib        face_voice_svm.joblib
fusion3_knn.joblib           fusion3_svm.joblib
command_knn.joblib           command_svm.joblib
```

Raw dimensions: face 8878, voice 30, fingerprint 8110. Training-only PCA dimensions:
57, 16, 13. Fusion classifier dimensions: 73 and 86. Both algorithms are exactly
KNN with n_neighbors=3 and SVC with a linear kernel. Scalers/PCA never fit test or
validation rows. Model and preprocessing provenance hashes are checked at load.

Selection used validation macro-F1, then accuracy, then KNN on ties. The selected
models are SVM for all five identity/fusion tasks and KNN for command recognition.
Identity thresholds minimize validation claimed-identity FAR+FRR. Command
thresholds minimize validation supported/UNKNOWN FAR+FRR using the exact runtime
gate. Training and inference share per-sample float32 transforms.

The initial experiment is preserved in results/initial_run.zip. A subsequent
independent source review found a command-calibration objective mismatch and an
action-scoring omission. These were corrected using validation data and an
arithmetic scoring test, not test-driven threshold tuning. The following rerun
uses the unchanged already-seen test set as a software regression check, not a
fresh blind performance estimate. No sample, model family or hyperparameter was
selected to improve a held-out result.

## Both Algorithms: Real Held-Out Results

Accuracy/precision/recall/F1 below are classification scores. FAR/FRR additionally
apply the claimed-identity prediction AND distance rule. Command rates instead
measure unsupported-word acceptance and supported-word rejection.

| Task | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | FAR | FRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Face | KNN | 0.3333 | 0.1111 | 0.3333 | 0.1667 | 0.0000 | 0.8750 |
| Face | SVM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0417 |
| Voice | KNN | 0.8889 | 0.9048 | 0.8889 | 0.8821 | 0.0556 | 0.1111 |
| Voice | SVM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| Fingerprint | KNN | 0.6667 | 0.5000 | 0.6667 | 0.5556 | 0.0000 | 0.5000 |
| Fingerprint | SVM | 0.8333 | 0.8889 | 0.8333 | 0.8222 | 0.0000 | 0.5000 |
| Command | KNN | 0.5556 | 0.6400 | 0.5333 | 0.5233 | 0.3333 | 0.0000 |
| Command | SVM | 0.3889 | 0.2467 | 0.3333 | 0.2810 | 0.3333 | 0.5000 |
| Face+voice | KNN | 0.3889 | 0.4510 | 0.3889 | 0.2692 | 0.3056 | 0.6111 |
| Face+voice | SVM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| Three modalities | KNN | 0.5000 | 0.4667 | 0.5000 | 0.4127 | 0.1667 | 0.8333 |
| Three modalities | SVM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 |

The biometric genuine/impostor denominators are face 24/48, voice 18/36,
fingerprint 6/12, face+voice 18/36, fusion3 6/12. Command support rejection uses
12 supported and 6 unsupported test utterances. These counts are recorded with
numerators in results/metrics/test_metrics.csv. Binary verification precision,
recall, F1 and accuracy are also saved, not substituted with identity scores.

Zero observed FAR in a tiny enrolled-person test is not a claim about unseen
attackers. Nonzero false acceptance exists for several KNN variants and for
unsupported words. The apparently perfect identification rows must not obscure
the high genuine rejection rate or face-image dependence.

## Full Hierarchy Evaluation

Six distinct held-out source triplets, tried with each of three claims, give
6 genuine + 12 impostor claim attempts. Repeated claims are dependent trials.

| Variant | Genuine Accepted | Impostor Accepted | Claim Accuracy | Macro F1 | FAR | FRR | Simulated Actions | Unexpected vs Source Word | Permission Violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All KNN | 0/6 | 0/12 | 0.6667 | 0.4000 | 0.0000 | 1.0000 | 0 | 0 | 0 |
| All SVM | 3/6 | 0/12 | 0.8333 | 0.7778 | 0.0000 | 0.5000 | 1 | 0 | 0 |
| Validation-selected | 3/6 | 0/12 | 0.8333 | 0.7778 | 0.0000 | 0.5000 | 2 | 2 | 0 |

Selected binary genuine-acceptance precision=1, recall=0.5 and F1=0.6667. The
selected Boolean execute/deny accuracy is 0.7222, but exact intended-action
accuracy is only 0.6667. Both executed actions used the wrong word: Employee GO
became STOP, and Guest GO became STOP. The first is a wrong-but-permitted action;
the second should have been denied based on its real source word. Both are
preserved as safety limitations, not described as correct behavior.
No actual equipment is controlled.

## Complete Prepared Trace: Kareem

Claim: user_02 / kareem. Source files: `face_02.png`,
`off/8e05039f_nohash_3.wav`, DB1_B `102_1`. All originate in the test partition.
The demo face.jpg is a JPEG conversion, not an extra independent observation.

| Stage | Status | Predicted | Claimed | Distance | Fixed Threshold |
| --- | --- | --- | --- | ---: | ---: |
| Face | ACCEPTED | user_02 | user_02 | 0.341503 | 0.754957 |
| Speaker | ACCEPTED | user_02 | user_02 | 0.823147 | 1.476382 |
| Face+voice | ACCEPTED | user_02 | user_02 | 0.888424 | 1.563036 |
| Fingerprint | ACCEPTED | user_02 | user_02 | 0.423691 | 0.825633 |
| Fusion3 | ACCEPTED | user_02 | user_02 | 1.252052 | 1.555737 |

The same voice vector then predicts **ON**, although the source word is **OFF**.
Command distance=1.013523; fixed threshold=1.303286. Verified role=Employee;
ON permission=DENIED; final result=**ACCESS DENIED**; action=None; simulated
state unchanged. This demonstrates both genuine biometric inference and a real
command-recognition failure; it is not a successful OFF command demonstration.

Abdelrahman: fingerprint distance=1.241882 > 0.825633; fusion3/command NOT RUN.
Renad: fingerprint distance=1.125656 > 0.825633; fusion3/command NOT RUN.
The complete five-stage dictionaries and explanations for every prepared person
are saved in results/metrics/demo_traces.json. Larger genuine/impostor traces
are in results/metrics/hierarchy_traces.json. Files were fixed before testing.

## Demo Verification and Tests

- Real Streamlit AppTest loaded each of the three held-out folders and invoked
  saved models. No retraining occurs at Verify.
- 31 automated tests passed, zero skipped. Routing doubles were used only for
  control-flow tests, not biometric results. Real tests covered wrong claims,
  corruption, data hashes, mapping, train-only preprocessing, all twelve models,
  no-reuse fusion sources and UI camera/impostor states. They also check actual
  runtime validation FAR/FRR for all twelve variants and exact-command scoring.
  The JUnit result is saved in results/metrics/tests.xml.
- Main notebook: 16 code cells executed without errors. App notebook: 5 code
  cells executed without errors. Actual outputs are saved in the notebook JSON.
- The served browser page at http://127.0.0.1:8501 was exercised, not just imported.
  The face/fingerprint images decoded, audio control was present, Kareem's real
  DENIED result appeared, and an impostor claim produced VERIFICATION FAILED.
- Desktop 1440x1100 and mobile 390x1100 screenshots were visually inspected in
  VS Code. Text wrapped, images loaded and document width stayed 390 on mobile.
- Real camera control displayed a permission request, with Take Photo disabled.
  Verify was also disabled without capture. No genuine camera image was acquired;
  live model inference is therefore not verified. Offline mode remained usable.

## Bonus Compliance

| Bonus | Implemented | Actually Demonstrated | Remaining Limitation |
| --- | --- | --- | --- |
| 1: verification-based speech commands | Yes | Real biometric, recognition and permission paths exercised | Poor speech accuracy; two wrong-word actions observed |
| 2: real-time verification | Genuine camera-input path | No genuine capture; permission-blocked | Allow real webcam, capture and run actual inference before claiming complete |
| 3: custom data | Supplied custom face data | Yes, prepared and trained | Team-collected voice NOT SATISFIED; public voice used |
| 4: fingerprints | Real DB1_B models and fusion | Trained/evaluated, true source impressions | Genuine FRR 50%; tiny validation/test size |

## Exact Run Commands

Use the two notebooks with the project .venv kernel, or execute:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe dataset_setup.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

To finish the live bonus: open localhost in a camera-enabled browser, allow its
real camera, select Live Face + Offline Voice + Offline Fingerprint, take a new
snapshot, then Verify. Do not call all modalities live. Improving accuracy would
require a new, declared experiment with more independent consented data, not
editing these test outcomes. Source/manual-download details are in DATASET_SOURCES.md;
English and Egyptian Arabic module explanations are in ORAL_DEFENSE.md.