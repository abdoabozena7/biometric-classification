# Dataset Sources and Provenance

## Supplied Faces

Original folder: `datasets/images/`. Three discovered folders: Abdelrahman, kareem,
renad; 40 PNG files each. Originals are never overwritten. All 120 decode;
no exact file or decoded-image duplicates were accepted. Similar-image warnings
are preserved in data/manifests/face_quality.json. No capture-session metadata
was supplied, so independent acquisition cannot be established. Do not equate
artificial variations or nearby frames with new independent people/sessions.
No facial recognition embedding, neural detector or cloud service is used.

## Speech Commands v0.02

- Author/source: Google / Pete Warden, Speech Commands v0.02.
- Official description: https://www.tensorflow.org/datasets/catalog/speech_commands
- Original archive: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
- Distribution used: https://huggingface.co/datasets/pollen-robotics/speech-commands-v0.02
- Pinned mirror revision: `278418261b4161f6603e06431fa05f9416aaee99`.
- Declared dataset license: Creative Commons Attribution 4.0, as recorded on the
  mirror card. Preserve attribution and review the original terms before reuse.
- Paper: Warden, P. (2018), Speech Commands: A Dataset for Limited-Vocabulary
  Speech Recognition, https://arxiv.org/abs/1804.03209 .

The mirror's original-train metadata contains 84,849 rows in 43 Parquet shards.
The setup reads only filename, word, speaker and utterance metadata first using
HTTP range reads. It then reads audio row groups needed for selected candidate
speakers, retaining genuine WAVs only. Parquet row-group granularity transfers
some unselected bytes; this is not a claim that only 90 WAVs' worth of bandwidth
was used. The complete 2.37-GiB source archive was not downloaded.

The viewer's per-recording API returned HTTP 429; the final implementation does
not depend on it. It reads pinned Parquet source rows and checks filenames.

Selected original speakers: **834f03fe**, **8e05039f**, **3bfd30e6**. Each has five
distinct decoded recordings for each ON/OFF/GO/STOP/YES/NO. A previous candidate,
b5cf6ea8, was rejected for duplicated audio content, not for model performance.
The complete candidate audit and real source identifiers remain in the manifests.

Each speaker contributes 30 utterances: 18 train, 6 validation, 6 test. Across
three speakers: 90 utterances, with 60 supported and 30 unsupported examples.
YES/NO are mapped to UNKNOWN only for the command target; their actual words
remain in samples.csv. Every source filename retains its speaker ID and
utterance index. All prepared audio is mono, 16-kHz PCM WAV.

All three enrolled speakers occur in every local partition, but no utterance or
decoded audio content crosses partitions. This is intentionally a local
utterance-level experiment and is **not the official speaker-disjoint benchmark**.

## FVC2004 DB1_B

- Official page tried first: https://bias.csr.unibo.it/fvc2004/download.asp
- Reachable official page: http://bias.csr.unibo.it/fvc2004/download.asp
- Actual archive: http://bias.csr.unibo.it/fvc2004/Downloads/DB1_B.zip
- Database description: http://bias.csr.unibo.it/fvc2004/databases.asp
- Collection: genuine optical fingerprint captures, DB1_B, 10 fingers x 8
  impressions. The locally selected fingers are **101, 102, 103**.
- DB4_B is synthetic; it is not downloaded or represented as a real capture.

The HTTPS site failed certificate hostname validation in this runtime. Its
advertised HTTP page/archive returned HTTP 200; no certificate checks were
disabled. HTTP has no transport authenticity guarantee. The downloaded ZIP is
validated, and its SHA-256 and exact source URL are recorded in
data/manifests/fingerprint_download.json. This hash records the received bytes;
it is not a publisher-signed authenticity proof.

Selected files preserve original archive member names, original finger IDs and
impression numbers such as `101_1`. PNG conversion affects prepared copies only.
Each finger uses five training, one validation and two test impressions, all
distinct decoded images. Total selected: 24 genuine impressions, split 15/3/6.

The official site publicly provides these B sample ZIPs; it does not establish an
unrestricted redistribution license. Use locally for this academic exercise,
review the provider's current terms and seek permission where required. Do not
publish the archive, biometric images or derived identifying models. No gated
access, account requirement or license agreement was bypassed.

## Virtual Identity Mapping

| Virtual ID | Face | Original Speaker | Original Finger |
| --- | --- | --- | --- |
| user_01 | Abdelrahman | 834f03fe | FVC2004_DB1_B:101 |
| user_02 | kareem | 8e05039f | FVC2004_DB1_B:102 |
| user_03 | renad | 3bfd30e6 | FVC2004_DB1_B:103 |

These are unrelated physical subjects. The mapping is an explicit educational
association, never a claim of authentic same-person multimodal data. Full physical
identity verification would require consented collection of all modalities from
each same person and enough independent sessions.

## Exact Manual Recovery

Preserve the existing data/raw cache if possible. With that cache, run:

```powershell
.\.venv\Scripts\python.exe dataset_setup.py --local-only
```

If the speech mirror is unavailable and the metadata cache is missing:

1. Download the original archive from the Speech Commands URL above, subject to
   its CC BY attribution requirements. This fallback is a larger download.
2. Extract its genuine `on`, `off`, `go`, `stop`, `yes`, `no` directories under
   `data/raw/speech_commands/`. Preserve names such as `on/834f03fe_nohash_3.wav`.
   Keep all candidate recordings in those directories; five different filenames
   are insufficient if they encode identical audio.
3. Run `python dataset_setup.py --local-only`. With no metadata cache, the script
   parses speaker/utterance IDs from the authentic source filenames. The original
   benchmark partition is honestly marked unrecorded in this manual path. The
   local train/validation/test assignment is still explicit and content-checked.
4. A different available archive subset may select different speakers, so review
   the generated mapping and retrain/evaluate. Never reuse old model artifacts
   against a changed mapping or change samples to improve an observed test score.

If fingerprints are unavailable, download the official **DB1_B.zip** and place it
at `data/raw/DB1_B.zip`; run the same `--local-only` command. The script checks
the ZIP, image decoding, finger IDs, impression counts and content uniqueness.
Do not substitute DB4_B or generated fingerprint images.

If fewer than three qualifying speakers/fingers exist or files cannot be obtained
under acceptable terms, setup reports BLOCKED in data/manifests/setup_status.json.
Face-only diagnostics remain available, but missing modalities are never silently
bypassed. Run `train.py` and `evaluate.py` only for the available real data, and
do not describe a PARTIAL model set as the complete system.