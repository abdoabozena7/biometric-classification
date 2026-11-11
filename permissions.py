"""Authorization is a dictionary lookup, not a machine-learning model."""

from config import ROLES

PERMISSIONS = {"Admin": {"ON", "OFF", "GO", "STOP"},
               "Employee": {"OFF", "GO", "STOP"},
               "Guest": {"OFF", "STOP"}}


def check_permission(verified_user_id, recognized_command):
    role = ROLES.get(verified_user_id)
    return "ALLOWED" if recognized_command in PERMISSIONS.get(role, set()) else "DENIED"