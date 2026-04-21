from __future__ import annotations

import pathlib


def stage2_package_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def project_root() -> pathlib.Path:
    """Repository root (parent of ``stage2/``)."""
    return stage2_package_dir().parent


def ours_package_dir() -> pathlib.Path:
    """Directory containing ``FS_method.py`` (``OURS/``)."""
    return project_root() / "OURS"
