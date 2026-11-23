"""Deliberately insecure demo code for validating code-xplorer pipelines.

Every issue in this file is intentional, planted ground truth.
See demo/README.md for the scoring checklist. DO NOT copy into real code.
"""
import sqlite3
import subprocess


def get_user(conn, user_id):
    """Planted: SQL injection via string formatting."""
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = '%s'" % user_id)
    return cur.fetchone()


def run_backup(path):
    """Planted: shell injection via shell=True + string concatenation."""
    subprocess.call("tar -czf backup.tgz " + path, shell=True)
