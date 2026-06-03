from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from sqlalchemy.orm import Session, sessionmaker

from db import run_with_retry, session_scope


T = TypeVar('T')


class BaseRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def _execute(self, fn: Callable[[Session], T]) -> T:
        def operation() -> T:
            with session_scope(self.session_factory) as session:
                return fn(session)
        return run_with_retry(operation)
