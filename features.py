"""Small, traditional feature extractors shared by notebooks and the demo."""

import hashlib
import io

import librosa
import numpy as np
import soundfile as sf
from PIL import Image, ImageOps
from skimage.feature import hog, local_binary_pattern


def load_image(source, mode="RGB", size=128):
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert(mode)
        if min(image.size) < 16:
            raise ValueError("Image is too small to contain useful biometric detail.")
        image = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
        return np.asarray(image)


def hog_lbp(gray):
    hog_vector = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), feature_vector=True)
    lbp_image = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_histogram = np.histogram(lbp_image, bins=np.arange(11), range=(0, 10))[0]
    lbp_histogram = lbp_histogram / max(lbp_histogram.sum(), 1)
    return np.concatenate([hog_vector, lbp_histogram]).astype(np.float32)


def face_features(source):
    rgb = load_image(source)
    gray = np.asarray(Image.fromarray(rgb).convert("L"))
    if gray.std() < 2:
        raise ValueError("Face image has almost no visible detail.")
    histograms = [np.histogram(rgb[:, :, channel], bins=256, range=(0, 256))[0]
                  / rgb.shape[0] / rgb.shape[1] for channel in range(3)]
    return np.concatenate([hog_lbp(gray), *histograms]).astype(np.float32)


def fingerprint_features(source):
    gray = load_image(source, mode="L")
    if gray.std() < 2:
        raise ValueError("Fingerprint image has almost no visible detail.")
    return hog_lbp(gray)


def load_audio(source):
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    samples, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if not np.isfinite(samples).all() or len(samples) < sample_rate * 0.15:
        raise ValueError("Audio is corrupt or shorter than 0.15 seconds.")
    if len(samples) > sample_rate * 10:
        raise ValueError("Use an isolated word, not audio longer than 10 seconds.")
    if np.sqrt(np.mean(samples ** 2)) < 0.0001:
        raise ValueError("Audio is silent or too quiet.")
    if sample_rate != 16000:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
    return samples.astype(np.float32)


def voice_features(source):
    samples = load_audio(source)
    mfcc = librosa.feature.mfcc(y=samples, sr=16000, n_mfcc=13,
                                n_fft=512, hop_length=160)
    zcr = librosa.feature.zero_crossing_rate(samples, frame_length=512, hop_length=160)
    rms = librosa.feature.rms(y=samples, frame_length=512, hop_length=160)
    vector = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1),
                             zcr.mean(axis=1), zcr.std(axis=1),
                             rms.mean(axis=1), rms.std(axis=1)])
    if vector.shape != (30,) or not np.isfinite(vector).all():
        raise ValueError("Audio did not produce 30 finite traditional features.")
    return vector.astype(np.float32)


def image_signature(source):
    gray = load_image(source, mode="L", size=9)
    bits = (gray[:, 1:] > gray[:, :-1]).flatten()
    return np.packbits(bits).tobytes().hex()


def decoded_hash(source, modality):
    if modality == "voice":
        content = load_audio(source).tobytes()
    else:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            content = str(image.size).encode("ascii") + image.tobytes()
    return hashlib.sha256(content).hexdigest()