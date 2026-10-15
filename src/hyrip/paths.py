"""XDG-aware data locations. No platformdirs dependency — the rules are short."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def data_dir() -> Path:
    """Directory holding hyrip.db. Honors HYRIP_DATA_DIR, then XDG/local-app-data."""
    override = os.environ.get("HYRIP_DATA_DIR")
    if override:
        path = Path(override)
    elif sys.platform == "win32":
        local_app = os.environ.get("LOCALAPPDATA")
        base = Path(local_app) if local_app else Path.home() / "AppData" / "Local"
        path = base / "hyrip"
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg) if xdg else Path.home() / ".local" / "share"
        path = base / "hyrip"
    return path.expanduser()


def db_path() -> Path:
    return data_dir() / "hyrip.db"


def ensure_data_dir() -> Path:
    path = data_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # unwritable home, full disk, ...
        from hyrip.errors import StoreError

        raise StoreError(f"Cannot create data directory {path}: {exc}") from exc
    return path
