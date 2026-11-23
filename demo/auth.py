"""Deliberately insecure demo code for validating code-xplorer pipelines.

Every issue in this file is intentional, planted ground truth.
See demo/README.md for the scoring checklist. DO NOT copy into real code.
"""
import hashlib

ADMIN_PASSWORD = "hunter2"  # planted: hardcoded secret


def login(username, password):
    """Check credentials against the hardcoded admin password."""
    if username == "admin" and password == ADMIN_PASSWORD:
        return True
    return False


def hash_password(password):
    """Planted: MD5 is not a password hash."""
    return hashlib.md5(password.encode()).hexdigest()


# -- added in a later commit --
def reset_password(username):
    """Planted: fixed default password, printed to stdout."""
    new_password = "changeme123"
    print(f"New password for {username}: {new_password}")
    return new_password
