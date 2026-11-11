"""Camera-first sign-in using the saved biometric verifiers."""

import streamlit as st

from camera_face import CAMERA_MODEL, load_camera_system, verify_camera_face
from config import MODELS
from verification import load_system, verify_fingerprint, verify_voice


@st.cache_resource
def cached_system(index_modified, camera_modified):
    system = load_system()
    return system, load_camera_system(system["identities"])


def check_sample(stage, verifier, sample, claim, system, attempt):
    """Reuse a decision only for the same upload and preceding steps."""
    key = f"{stage}_check"
    input_key = (attempt, sample.file_id)
    previous = st.session_state.get(key)
    if previous is None or previous[0] != input_key:
        with st.spinner(f"Checking {stage}…"):
            decision = verifier(sample.getvalue(), claim, system)
        st.session_state[key] = (input_key, decision)
    return st.session_state[key][1]


st.set_page_config(page_title="Biometric Sign-in", layout="centered")
st.title("Biometric Sign-in")

try:
    model_revision = (MODELS / "model_index.json").stat().st_mtime_ns
    camera_revision = CAMERA_MODEL.stat().st_mtime_ns
    system, camera = cached_system(model_revision, camera_revision)
except FileNotFoundError:
    st.error("Saved models are missing. Run python train.py and python camera_face.py.")
    st.stop()
except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
    st.error(f"Sign-in is unavailable. Could not load the saved models: {error}")
    st.stop()

identities = system["identities"]
claim = st.selectbox(
    "Account", list(identities), index=None, placeholder="Choose your name",
    format_func=lambda user_id: identities[user_id]["display_name"].title(),
)
if claim is None:
    st.stop()

folder = identities[claim]["display_name"]
name = folder.title()
attempt = f"{claim}_{model_revision}_{camera_revision}"

st.subheader("1. Face")
st.caption("Face the camera and center your face, then take a photo.")
photo = st.camera_input("Face photo", label_visibility="collapsed", width=420, key=f"photo_{attempt}")
if photo is None:
    st.stop()

face = check_sample("face", verify_camera_face, photo, claim, camera, attempt)
if not face["accepted"]:
    message = f"Could not verify your face for {name}. Clear the photo and try again."
    if face["predicted_identity"] is None:
        message = face["explanation"]
    st.error(message)
    st.stop()
st.success("Face verified.", icon=":material/check_circle:")

face_attempt = f"{attempt}_{photo.file_id}"
st.subheader("2. Voice")
st.caption(f"Choose your WAV recording from `datasets/demo/{folder}`.")
voice = st.file_uploader("Voice recording", type=["wav"], key=f"voice_{face_attempt}")
if voice is None:
    st.stop()

speaker = check_sample("voice", verify_voice, voice, claim, system, face_attempt)
if not speaker["accepted"]:
    message = "Choose a readable WAV recording with a clear, short utterance."
    if speaker["predicted_identity"] is not None:
        message = f"Voice did not verify for {name}. Choose another recording."
    st.error(message)
    st.stop()
st.success("Voice verified.", icon=":material/check_circle:")

voice_attempt = f"{face_attempt}_{voice.file_id}"
st.subheader("3. Fingerprint")
st.caption(f"Choose your fingerprint image from `datasets/demo/{folder}`.")
fingerprint = st.file_uploader(
    "Fingerprint image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    key=f"fingerprint_{voice_attempt}",
)
if fingerprint is None:
    st.stop()

finger = check_sample("fingerprint", verify_fingerprint, fingerprint, claim, system, voice_attempt)
if not finger["accepted"]:
    message = "Choose a readable fingerprint image."
    if finger["predicted_identity"] is not None:
        message = f"Fingerprint did not verify for {name}. Choose another image."
    st.error(message)
    st.stop()

st.success(f"Signed in as {name}.", icon=":material/check_circle:")
