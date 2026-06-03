from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from models import ProactiveNotification, now_utc
from repositories.base import BaseRepository


class ProactiveNotificationsRepository(BaseRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        super().__init__(session_factory)

    def exists_recent(self, *, user_id: str, dedupe_key: str, hours: int = 24) -> bool:
        cutoff = now_utc() - timedelta(hours=hours)
        def operation(session: Session) -> bool:
            stmt = select(ProactiveNotification).where(
                ProactiveNotification.user_id == user_id,
                ProactiveNotification.dedupe_key == dedupe_key,
                ProactiveNotification.last_sent_at >= cutoff,
            )
            return session.scalar(stmt) is not None
        return self._execute(operation)

    def record(self, *, user_id: str, notification_type: str, message: str, dedupe_key: str, sent_at: datetime | None = None) -> ProactiveNotification:
        sent = sent_at or now_utc()
        def operation(session: Session) -> ProactiveNotification:
            notification = ProactiveNotification(
                user_id=user_id,
                notification_type=notification_type,
                message=message,
                dedupe_key=dedupe_key,
                last_sent_at=sent,
            )
            session.add(notification)
            session.flush()
            return notification
        return self._execute(operation)

    def latest_for_user(
        self,
        *,
        user_id: str,
        notification_type: str | None = None,
    ) -> ProactiveNotification | None:
        def operation(session: Session) -> ProactiveNotification | None:
            stmt = select(ProactiveNotification).where(
                ProactiveNotification.user_id == user_id
            )
            if notification_type is not None:
                stmt = stmt.where(
                    ProactiveNotification.notification_type == notification_type
                )
            stmt = stmt.order_by(
                ProactiveNotification.last_sent_at.desc(),
                ProactiveNotification.id.desc(),
            ).limit(1)
            return session.scalar(stmt)

        return self._execute(operation)
