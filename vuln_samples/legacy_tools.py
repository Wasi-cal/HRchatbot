"""Intentionally vulnerable helpers for testing veil. Not imported by the app."""

import hashlib
import os
import pickle
import random
import sqlite3
import subprocess

import requests
import yaml

# 1. Hardcoded credentials (fake values)
DB_PASSWORD = "hr-admin-Passw0rd!"
API_KEY = "sk-test-0000000000000000000000000000"


def find_employee(db_path: str, name: str):
    # 2. SQL injection: user input concatenated into the query
    conn = sqlite3.connect(db_path)
    return conn.execute("SELECT * FROM employees WHERE name = '" + name + "'").fetchall()


def ping_host(host: str) -> str:
    # 3. OS command injection: shell=True with user input
    return subprocess.check_output("ping -c 1 " + host, shell=True, text=True)


def read_policy_file(filename: str) -> str:
    # 4. Path traversal: no check that the path stays inside the policies folder
    with open(os.path.join("policies", filename), encoding="utf-8") as f:
        return f.read()


def load_session(blob: bytes):
    # 5. Insecure deserialization of untrusted data
    return pickle.loads(blob)


def hash_password(password: str) -> str:
    # 6. Weak hash (MD5, unsalted) for passwords
    return hashlib.md5(password.encode()).hexdigest()


def make_reset_token() -> str:
    # 7. Predictable token from a non-cryptographic PRNG
    return str(random.randint(100000, 999999))


def fetch_url(url: str) -> str:
    # 8. SSRF (no allow-list) and disabled TLS verification
    return requests.get(url, verify=False, timeout=10).text


def parse_config(text: str):
    # 9. Unsafe YAML load can construct arbitrary objects
    return yaml.load(text, Loader=yaml.Loader)


def run_expression(expr: str):
    # 10. Code injection via eval on user input
    return eval(expr)
