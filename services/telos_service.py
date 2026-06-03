from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

UserId = Union[int, str]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
TELOS_TEMPLATE_PATH = _PROJECT_ROOT / "resources" / "telos_template.md"

_VALID_USER_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_user_id(user_id: UserId) -> str:
    safe = str(user_id).strip()
    if not safe or not _VALID_USER_ID.match(safe):
        raise ValueError(f"Invalid user_id for TELOS path: {user_id!r}")
    return safe


class TelosService:
    def __init__(self, telos_dir: Path) -> None:
        self._telos_dir = Path(telos_dir)
        self._telos_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._telos_dir.chmod(0o700)
        except (PermissionError, OSError):
            pass

    @property
    def telos_dir(self) -> Path:
        return self._telos_dir

    def path_for(self, user_id: UserId) -> Path:
        safe = _validate_user_id(user_id)
        return self._telos_dir / f"{safe}.md"

    def load(self, user_id: UserId) -> Optional[str]:
        p = self.path_for(user_id)
        if not p.exists():
            return None
        return p.read_text(encoding="utf-8")

    def has_telos(self, user_id: UserId) -> bool:
        return self.path_for(user_id).exists()

    def append(self, user_id: UserId, content: str) -> Path:
        p = self.path_for(user_id)
        with open(p, 'a', encoding='utf-8') as fh:
            fh.write(content)
        try:
            p.chmod(0o600)
        except (PermissionError, OSError):
            pass
        return p


def load_telos_template() -> str:
    return TELOS_TEMPLATE_PATH.read_text(encoding="utf-8")
