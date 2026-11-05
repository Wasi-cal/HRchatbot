"""Intentionally vulnerable helpers for testing veil. Not imported by the app."""

import hashlib
import json
import os
import random
import sqlite3
import subprocess

import requests
import yaml

# 1. Hardcoded credentials (fake values)
DB_PASSWORD = "hr-admin-Passw0rd!"
API_KEY = "sk-test-0000000000000000000000000000"


def find_employee(db_path: str, name: str):
    # 2. Use a parameterized query to prevent SQL injection
    conn = sqlite3.connect(db_path)
    return conn.execute("SELECT * FROM employees WHERE name = ?", (name,)).fetchall()


def ping_host(host: str) -> str:
    # 3. OS command injection: shell=True with user input
    return subprocess.check_output(["ping", "-c", "1", host], shell=False, text=True)


def read_policy_file(filename: str) -> str:
    # 4. Path traversal: no check that the path stays inside the policies folder
    with open(os.path.join("policies", filename), encoding="utf-8") as f:
        return f.read()


def load_session(blob: bytes):
    # 5. Parse session data without allowing code execution
    return json.loads(blob)


def hash_password(password: str) -> str:
    # 6. Weak hash (MD5, unsalted) for passwords
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"{salt.hex()}:{digest.hex()}"


def make_reset_token() -> str:
    # 7. Predictable token from a non-cryptographic PRNG
    return str(random.randint(100000, 999999))


def fetch_url(url: str) -> str:
    # 8. SSRF (no allow-list)
    return requests.get(url, timeout=10).text


def parse_config(text: str):
    # 9. Unsafe YAML load can construct arbitrary objects
    return yaml.load(text, Loader=yaml.Loader)


def run_expression(expr: str):
    # 10. Code injection via eval on user input
    return eval(expr)
