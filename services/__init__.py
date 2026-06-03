"""Nexus services package.

Some legacy service helpers currently ship only as compiled ``.pyc`` files in
``services/__pycache__/``. Register a tiny finder so imports such as
``services.approval_exemptions`` still resolve while the source files are
absent.
"""
from __future__ import annotations

from importlib.abc import MetaPathFinder
from importlib.machinery import SourcelessFileLoader
from importlib.util import spec_from_file_location
from pathlib import Path
import sys


_PACKAGE_NAME = __name__
_PACKAGE_DIR = Path(__file__).resolve().parent
_PYCACHE_DIR = _PACKAGE_DIR / "__pycache__"


class _ServicesPycacheFinder(MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        del path, target
        if not fullname.startswith(f"{_PACKAGE_NAME}."):
            return None
        short_name = fullname.rsplit(".", 1)[-1]
        if (_PACKAGE_DIR / f"{short_name}.py").exists():
            return None
        candidates = sorted(_PYCACHE_DIR.glob(f"{short_name}.cpython-*.pyc"))
        if not candidates:
            return None
        pyc_path = candidates[0]
        loader = SourcelessFileLoader(fullname, str(pyc_path))
        return spec_from_file_location(fullname, pyc_path, loader=loader)


def _install_pycache_finder() -> None:
    for finder in sys.meta_path:
        if isinstance(finder, _ServicesPycacheFinder):
            return
    sys.meta_path.insert(0, _ServicesPycacheFinder())


_install_pycache_finder()
