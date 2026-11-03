"""Helpers for employee lookups, policy files and config parsing. Not imported by the app."""

import hashlib
import ipaddress
import os
import secrets
import socket
import sqlite3
import subprocess
from urllib.parse import urlparse

import requests
import yaml

POLICIES_DIR = os.path.realpath("policies")
ALLOWED_URL_SCHEMES = {"http", "https"}


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


def db_password() -> str:
    return _required_env("DB_PASSWORD")


def api_key() -> str:
    return _required_env("API_KEY")


def find_employee(db_path: str, name: str):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT * FROM employees WHERE name = ?", (name,)).fetchall()
    finally:
        conn.close()


def ping_host(host: str) -> str:
    ipaddress.ip_address(host)
    return subprocess.check_output(["ping", "-c", "1", host], text=True)


def read_policy_file(filename: str) -> str:
    path = os.path.realpath(os.path.join(POLICIES_DIR, filename))
    if os.path.commonpath([POLICIES_DIR, path]) != POLICIES_DIR:
        raise ValueError("path is outside the policies folder")
    with open(path, encoding="utf-8") as f:
        return f.read()


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return salt.hex() + "$" + digest.hex()


def make_reset_token() -> str:
    return secrets.token_urlsafe(32)


def fetch_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_URL_SCHEMES or not parsed.hostname:
        raise ValueError("only http and https URLs are allowed")
    for info in socket.getaddrinfo(parsed.hostname, None):
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise ValueError("URL resolves to a non-public address")
    return requests.get(url, timeout=10, allow_redirects=False).text


def parse_config(text: str):
    return yaml.safe_load(text)


def run_expression(expr: str):
    import ast

    return ast.literal_eval(expr)
