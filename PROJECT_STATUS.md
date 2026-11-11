# Project Acceptance Log

The user's full attached specification governs the implementation. The latest
request changes the presentation to simple, explained notebooks; it does not
remove the data-integrity, model, verification, or reporting requirements.

## Task Graph

| Task | Depends On | Observable Outcome | Owner | Check |
| --- | --- | --- | --- | --- |
| A: real data | none | 3 face folders audited, 3 consistent speakers, 3 real fingers, honest virtual mapping, immutable splits, held-out live_test | main session | decode and content hashes, manifests, per-source counts |
| B: features and models | A | HOG/LBP/color, MFCC/ZCR/RMS, train-only scaling/PCA, 12 actual KNN/SVM artifacts | main session | feature dimensions, fitted pipelines, provenance and validation selection |
| C: protected pipeline | B | all 5 verification gates, same audio features, UNKNOWN rejection, role permissions, no actions after failure | main session | routing unit tests and actual held-out genuine/impostor runs |
| D: notebooks and UI | C | simple main notebook and separate Streamlit notebook, offline samples, honest camera mode, metrics | main session | clean notebook execution, UI interaction, screenshots |
| E: evaluation and explanation | A, B, C | real metrics, confusion matrices, FAR/FRR denominators, English/Egyptian Arabic oral guide and source terms | main session | saved evaluation artifacts, documented limitations and commands |
| F: final integration | A, B, C, D, E | original full goal checked, blockers retained, working user entrypoints | main session | pytest, actual notebook/model/data/UI checks and final report |

Work is serialized because the notebooks, modules, manifests and trained artifacts
share one implementation contract. A separate read-only Explore agent researched
the public sources and challenged the acceptance plan. Its live tool invocation is
in the current VS Code conversation; it did not download, edit or train anything.

## Independent Challenge

- Explicitly name all six tasks, with two fitted artifacts per task.
- Preserve the same utterance split for speaker and command tasks.
- Distinguish unsupported commands from unseen biometric identities.
- Hash decoded content as well as file bytes; flag near-duplicate faces.
- Freeze validation thresholds before inspecting held-out test outcomes.
- A prepared demo is a copy of a held-out test sample, not a new independent trial.
- Use one shared implementation for notebook, command line and UI.
- Never mark data/model/UI implementation as real execution or whole-goal success.

HarrTest v3 was loaded and used for acceptance-linked planning and real checks.
Its automatic host hook/root-gate enforcement has not been configured; this log
does not claim mechanically sealed or automatic whole-goal acceptance.

## Verified Offline Outcomes

- Supplied root: datasets/images, names Abdelrahman, kareem, renad.
- 40 files per person, 120 total; no identical file SHA-256 hashes.
- Sample images visually inspected; centered portraits suit simple preprocessing.
- Python 3.13.7 project virtual environment selected; dependencies installed.
- Official FVC HTTPS endpoint has a certificate hostname mismatch in this runtime.
- A: all 120 face files decode; 54/2/0 similar-pair warnings preserved. No capture
  session metadata exists, so acquisition independence remains unverified.
- A: selected real speakers 834f03fe, 8e05039f, 3bfd30e6; 90 distinct utterances.
  Candidate b5cf6ea8 rejected for duplicate content before any model training.
- A: official DB1_B acquired via advertised HTTP; real fingers101/102/103, 24
  distinct impressions. No synthetic fingerprint substitution or TLS bypass.
- A: deterministic manifests, no cross-split content duplication, pairs54/18/18,
  triplets15/3/6, three complete live_test folders; cached offline setup reproduced.
- B: all12 actual KNN(3)/linear SVM classifiers trained. Scalers/PCA fit training
  only. Raw/PCA dimensions face8878/57, voice30/16, fingerprint8110/13.
- C: actual five-gate hierarchy, shared voice features, UNKNOWN gate and dictionary
  authorization connected. Protected actions occur only after all gates/permission.
- D: both notebooks executed from clean project kernels (16 and5 code cells).
  Streamlit AppTest and served-browser offline/impostor states exercised. Desktop
  and mobile captures inspected; mobile document has no horizontal overflow.
- E: actual both-model metrics, verification denominators/confusion matrices,
  hierarchy and demo traces saved. Evaluation re-extracts real file features using
  frozen transforms; it does not use a mutable feature cache or fit on test rows.
- E: README, DATASET_SOURCES, bilingual ORAL_DEFENSE and FINAL_REPORT delivered.
- F:31 pytest tests passed without skips. A separate native Explore invocation
  researched sources and challenged the acceptance plan. Automatic HarrTest
  host-hook/root-seal enforcement is not claimed.
- Source review found command calibration and action-scoring definition issues;
  both were repaired and a fresh targeted review marked each resolved. The
  original models/results/source are retained in results/initial_run.zip.
  Validation calibration now exactly matches real runtime decisions for all12
  variants. This reused-test reevaluation is a software regression, not a new
  blind performance estimate; no held-out-driven tuning or sample replacement.

## Remaining Original-Goal Limitations

- Real-time bonus: camera path and no-capture safety implemented/tested; browser
  required permission and Take Photo was disabled. Genuine live capture/inference
  NOT VERIFIED. No offline image was passed off as a live frame.
- Team-collected voice bonus: NOT SATISFIED (public recordings intentionally used).
- Command selected test accuracy55.56%, unsupported acceptance2/6; two wrong-word
  simulated actions in selected hierarchy, zero permission-dictionary
  violations for the recognized word. This is not a production-safe recognizer.
- Fingerprint/fusion3 genuine FRR3/6; prepared demo files/thresholds were NOT
  manipulated. Abdelrahman/renad stop at fingerprint; kareem passes all biometric
  gates but OFF is predictedON and denied for Employee. No promised green demo.
- Same-person physical multimodal identity, real-world spoof resistance and
  statistically reliable open-set security were not established by this dataset.