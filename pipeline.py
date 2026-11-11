"""Face -> speaker -> early fusion -> fingerprint -> fusion3 -> word -> permission."""

from commands import execute_action, recognize_command
from config import ROLES
from fusion import concatenate_features
from permissions import check_permission
from verification import extract_vector, load_system, verify_vector

STAGES = ("face", "voice", "face_voice", "fingerprint", "fusion3")
STAGE_NAMES = {"face": "Face Verification", "voice": "Voice Verification",
               "face_voice": "Face + Voice Fusion Verification",
               "fingerprint": "Fingerprint Verification",
               "fusion3": "Three-Modality Fusion Verification"}


def run_pipeline(face_source, voice_source, fingerprint_source, claimed_identity,
                 system=None, state=None):
    state = state if state is not None else {"enabled": False, "running": False}
    trace = [{"task": task, "stage": STAGE_NAMES[task], "status": "NOT RUN",
              "accepted": False, "claimed_identity": claimed_identity,
              "predicted_identity": None, "distance": None, "threshold": None,
              "explanation": "An earlier stage has not passed."} for task in STAGES]
    result = {"result": "VERIFICATION FAILED", "authenticated": False,
              "claimed_identity": claimed_identity, "verified_identity": None,
              "role": None, "command": None, "permission": "NOT RUN", "action": None,
              "trace": trace, "state": state, "explanation": ""}
    try:
        system = system or load_system()
        if claimed_identity not in system["identities"]:
            raise ValueError("Claimed identity is not enrolled.")
    except Exception as error:
        result["explanation"] = f"Models/data unavailable: {type(error).__name__}: {error}"
        return result
    vectors = {}
    for stage in trace:
        task = stage["task"]
        try:
            if task == "face":
                vectors["face"] = extract_vector("face", face_source, system)
                vector = vectors["face"]
            elif task == "voice":
                vectors["voice"] = extract_vector("voice", voice_source, system)
                vector = vectors["voice"]
            elif task == "face_voice":
                vector = concatenate_features(vectors["face"], vectors["voice"])
            elif task == "fingerprint":
                vectors["fingerprint"] = extract_vector("fingerprint", fingerprint_source, system)
                vector = vectors["fingerprint"]
            else:
                vector = concatenate_features(vectors["face"], vectors["voice"], vectors["fingerprint"])
            decision = verify_vector(task, vector, claimed_identity, system)
            stage.update(decision)
        except Exception as error:
            stage["explanation"] = f"{type(error).__name__}: {error}"
        stage["status"] = "ACCEPTED" if stage["accepted"] else "REJECTED"
        if not stage["accepted"]:
            result["explanation"] = f"Stopped at {stage['stage']}: {stage['explanation']}"
            return result
    result.update(authenticated=True, verified_identity=claimed_identity, role=ROLES[claimed_identity])
    try:
        command = recognize_command(vectors["voice"], system)
        result["command"] = command
        if not command["supported"]:
            result.update(result="UNKNOWN COMMAND", explanation="The audio command was rejected. No action executed.")
            return result
        permission = check_permission(claimed_identity, command["command"])
        result["permission"] = permission
        if permission == "DENIED":
            result.update(result="ACCESS DENIED", explanation="The verified role is not permitted to execute this command.")
            return result
        result["action"] = execute_action(command["command"], state)
        result.update(result="ACCESS GRANTED", explanation=result["action"])
    except Exception as error:
        result.update(result="UNKNOWN COMMAND", explanation=f"Command processing failed safely: {error}")
    return result