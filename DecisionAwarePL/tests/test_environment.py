from __future__ import annotations

import importlib
import sys


def test_python_version_is_supported():
    assert sys.version_info.major == 3
    assert sys.version_info.minor == 11


def test_required_packages_import():
    modules = ["numpy", "scipy", "torch", "zuko", "matplotlib"]
    for name in modules:
        mod = importlib.import_module(name)
        assert mod is not None


