"""Literal machine defaults for reference commands run directly with Python."""

import os
from pathlib import Path


def local_setting(name):
    """An exported variable takes precedence over the checkout's .env.local."""
    if name in os.environ:
        return os.environ[name]
    path = Path(__file__).resolve().parents[2] / ".env.local"
    try:
        with path.open(encoding="utf-8") as config:
            for line in config:
                key, separator, value = line.rstrip("\r\n").partition("=")
                if separator and key == name:
                    return value
    except FileNotFoundError:
        pass
    return None
