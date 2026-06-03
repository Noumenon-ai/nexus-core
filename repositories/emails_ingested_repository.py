from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from models import EmailIngested
from repositories.base import BaseRepository


class EmailsIngestedRepository(BaseRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        super().__init__(session_factory)

    def upsert(self, *, user_id: str, gmail_message_id: str, subject: str | None, sender: str | None, snippet: str | None, received_at: datetime, category: str | None, extracted_json: str | None) -> EmailIngested:
        def operation(session: Session) -> EmailIngested:
            stmt = select(EmailIngested).where(EmailIngested.user_id == user_id, EmailIngested.gmail_message_id == gmail_message_id)
            email = session.scalar(stmt)
            if email is None:
                email = EmailIngested(
                    user_id=user_id,
                    gmail_message_id=gmail_message_id,
                    subject=subject,
                    sender=sender,
                    snippet=snippet,
                    received_at=received_at,
                    category=category,
                    extracted_json=extracted_json,
                )
                session.add(email)
            else:
                email.subject = subject
                email.sender = sender
                email.snippet = snippet
                email.received_at = received_at
                email.category = category
                email.extracted_json = extracted_json
            session.flush()
            return email
        return self._execute(operation)

    def list_recent(self, user_id: str, *, hours: int = 24) -> list[EmailIngested]:
        cutoff = datetime.now(received_tz()) - timedelta(hours=hours)
        def operation(session: Session) -> list[EmailIngested]:
            stmt = select(EmailIngested).where(EmailIngested.user_id == user_id, EmailIngested.received_at >= cutoff).order_by(EmailIngested.received_at.desc())
            return list(session.scalars(stmt).all())
        return self._execute(operation)


def received_tz():
    from datetime import timezone
    return timezone.utc
