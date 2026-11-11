"""Identify a person from the camera, then verify their voice and fingerprint."""

import streamlit as st

from camera_face import CAMERA_MODEL, load_camera_system
from camera_identification import identify_camera_face
from config import MODELS, ROOT
from verification import load_system, verify_fingerprint, verify_voice

STEPS = (("face", "Face"), ("voice", "Voice"), ("fingerprint", "Fingerprint"))


@st.cache_resource
def cached_system(revision):
    system = load_system()
    return system, load_camera_system(system["identities"])


def start_over():
    st.session_state.attempt = st.session_state.get("attempt", 0) + 1
    st.session_state.phase = "face"
    st.session_state.identity = None


def change_recording():
    st.session_state.phase = "voice"


def show_progress(phase):
    current = next((index for index, (stage, _) in enumerate(STEPS) if stage == phase), len(STEPS))
    items = []
    for index, (_, label) in enumerate(STEPS):
        if index < current:
            items.append(f'<li class="done">{label}<span class="sr-only"> complete</span></li>')
        elif index == current:
            items.append(f'<li aria-current="step">{label}</li>')
        else:
            items.append(f"<li>{label}</li>")
    st.html('<ol class="signin-steps" aria-label="Sign-in progress">' + "".join(items) + "</ol>")


def capture_face(camera):
    st.subheader("Take a photo")
    st.caption("Allow camera access and center your face in the frame.")
    photo = st.camera_input("Face photo", label_visibility="collapsed", key=f"photo_{st.session_state.attempt}")
    if photo is None:
        return
    with st.spinner("Identifying face…"):
        identity, error = identify_camera_face(photo.getvalue(), camera)
    if error:
        st.error(error)
        return
    st.session_state.identity = identity
    st.session_state.phase = "voice"
    st.rerun()


def verify_recording(identity, name, folder, system):
    st.subheader("Verify your voice")
    st.success(f"Identified as {name}.", icon=":material/check_circle:")
    st.caption(f"Upload `voice.wav` from `datasets/demo/{folder}`.")
    voice = st.file_uploader("Voice recording", type=["wav"], key=f"voice_{st.session_state.attempt}")
    if voice is None:
        return
    with st.spinner("Checking voice…"):
        decision = verify_voice(voice.getvalue(), identity, system)
    if decision["accepted"]:
        st.session_state.phase = "fingerprint"
        st.rerun()
    if decision["predicted_identity"] is None:
        st.error("Choose a readable WAV recording with a clear, short utterance.")
    else:
        st.error(f"Voice did not verify for {name}. Choose another recording.")


def verify_impression(identity, name, folder, system):
    st.subheader("Verify your fingerprint")
    st.success(f"Voice verified for {name}.", icon=":material/check_circle:")
    st.caption(f"Upload `fingerprint.png` from `datasets/demo/{folder}`.")
    fingerprint = st.file_uploader(
        "Fingerprint image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key=f"fingerprint_{st.session_state.attempt}",
    )
    if fingerprint is None:
        return
    with st.spinner("Checking fingerprint…"):
        decision = verify_fingerprint(fingerprint.getvalue(), identity, system)
    if decision["accepted"]:
        st.session_state.phase = "complete"
        st.rerun()
    if decision["predicted_identity"] is None:
        st.error("Choose a readable fingerprint image.")
    else:
        st.error(f"Fingerprint did not verify for {name}. Choose another image.")


st.set_page_config(page_title="Biometric Sign-in", layout="centered")
st.html(ROOT / "signin.css")
st.title("Biometric Sign-in")

try:
    revision = ((MODELS / "model_index.json").stat().st_mtime_ns, CAMERA_MODEL.stat().st_mtime_ns)
    system, camera = cached_system(revision)
except FileNotFoundError:
    st.error("Saved models or model metadata are missing. Restore the project files and try again.")
    st.stop()
except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
    st.error(f"Sign-in is unavailable. Could not load the saved models: {error}")
    st.stop()

if st.session_state.get("model_revision") != revision:
    start_over()
    st.session_state.model_revision = revision

phase = st.session_state.phase
show_progress(phase)

with st.container(key=f"signin-{phase}", border=True):
    if phase == "face":
        capture_face(camera)
    else:
        identity = st.session_state.identity
        folder = system["identities"][identity]["display_name"]
        name = folder.title()
        if phase == "voice":
            verify_recording(identity, name, folder, system)
        elif phase == "fingerprint":
            verify_impression(identity, name, folder, system)
        else:
            st.subheader("Verification complete")
            st.success(f"Signed in as {name}.", icon=":material/check_circle:")

        with st.container(horizontal=True):
            if phase == "fingerprint":
                st.button("Change recording", on_click=change_recording)
            st.button("Start over", on_click=start_over)
