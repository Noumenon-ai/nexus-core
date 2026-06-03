from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from models import Reminder, now_utc
from repositories.base import BaseRepository


def _ensure_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class RemindersRepository(BaseRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        super().__init__(session_factory)

    def _coerce(self, reminder: Reminder | None) -> Reminder | None:
        if reminder is None:
            return None
        reminder.next_fire_at = _ensure_utc(reminder.next_fire_at)
        reminder.last_fired_at = _ensure_utc(reminder.last_fired_at)
        reminder.last_failure_alert_at = _ensure_utc(
            reminder.last_failure_alert_at,
        )
        reminder.delivery_attempts = int(getattr(reminder, 'delivery_attempts', 0) or 0)
        return reminder

    def create(self, *, user_id: str, body: str, next_fire_at: datetime, recurrence: str | None, status: str = 'active') -> Reminder:
        def operation(session: Session) -> Reminder:
            reminder = Reminder(user_id=user_id, body=body, next_fire_at=next_fire_at, recurrence=recurrence, status=status)
            session.add(reminder)
            session.flush()
            return self._coerce(reminder)
        return self._execute(operation)

    def get_by_id(self, reminder_id: str) -> Reminder | None:
        return self._execute(lambda session: self._coerce(session.get(Reminder, reminder_id)))

    def list_active(self, user_id: str) -> list[Reminder]:
        def operation(session: Session) -> list[Reminder]:
            stmt = select(Reminder).where(Reminder.user_id == user_id, Reminder.status == 'active').order_by(Reminder.next_fire_at)
            return [self._coerce(reminder) for reminder in session.scalars(stmt).all()]
        return self._execute(operation)

    def list_all_active(self) -> list[Reminder]:
        def operation(session: Session) -> list[Reminder]:
            stmt = select(Reminder).where(Reminder.status == 'active').order_by(Reminder.next_fire_at)
            return [self._coerce(reminder) for reminder in session.scalars(stmt).all()]
        return self._execute(operation)

    def list_due_within(self, user_id: str, *, minutes: int) -> list[Reminder]:
        now = now_utc()
        upper = now + timedelta(minutes=minutes)
        def operation(session: Session) -> list[Reminder]:
            stmt = select(Reminder).where(Reminder.user_id == user_id, Reminder.status == 'active', Reminder.next_fire_at >= now, Reminder.next_fire_at <= upper).order_by(Reminder.next_fire_at)
            return [self._coerce(reminder) for reminder in session.scalars(stmt).all()]
        return self._execute(operation)

    def cancel(self, reminder_id: str) -> Reminder | None:
        def operation(session: Session) -> Reminder | None:
            reminder = session.get(Reminder, reminder_id)
            if reminder is None:
                return None
            reminder.status = 'cancelled'
            reminder.updated_at = now_utc()
            session.flush()
            return self._coerce(reminder)
        return self._execute(operation)

    def find_for_cancel(self, user_id: str, query: str) -> Reminder | None:
        query_lower = query.lower().strip()
        if not query_lower:
            return None
        def operation(session: Session) -> Reminder | None:
            stmt = select(Reminder).where(Reminder.user_id == user_id, Reminder.status == 'active').order_by(Reminder.next_fire_at)
            reminders = [self._coerce(reminder) for reminder in session.scalars(stmt).all()]
            for reminder in reminders:
                if reminder.id == query:
                    return reminder
                if query_lower in reminder.body.lower():
                    return reminder
            return reminders[0] if len(reminders) == 1 else None
        return self._execute(operation)

    def mark_fired(self, reminder_id: str, *, fired_at: datetime, next_fire_at: datetime | None = None, status: str | None = None) -> Reminder:
        def operation(session: Session) -> Reminder:
            reminder = session.get(Reminder, reminder_id)
            if reminder is None:
                raise ValueError('Reminder not found')
            reminder.last_fired_at = fired_at
            if next_fire_at is not None:
                reminder.next_fire_at = next_fire_at
            if status is not None:
                reminder.status = status
            reminder.updated_at = fired_at
            session.flush()
            return self._coerce(reminder)
        return self._execute(operation)

    def claim_fire(self, reminder_id: str, *, expected_next_fire_at: datetime, fired_at: datetime, next_fire_at: datetime | None = None, status: str | None = None) -> Reminder | None:
        def operation(session: Session) -> Reminder | None:
            values = {'last_fired_at': fired_at, 'updated_at': fired_at}
            if next_fire_at is not None:
                values['next_fire_at'] = next_fire_at
            if status is not None:
                values['status'] = status
            stmt = (
                update(Reminder)
                .where(
                    Reminder.id == reminder_id,
                    Reminder.status == 'active',
                    Reminder.next_fire_at == expected_next_fire_at,
                    or_(Reminder.last_fired_at.is_(None), Reminder.last_fired_at < expected_next_fire_at),
                )
                .values(**values)
            )
            result = session.execute(stmt)
            if result.rowcount == 0:
                return None
            session.flush()
            return self._coerce(session.get(Reminder, reminder_id))
        return self._execute(operation)

    def mark_delivery(self, reminder_id: str, *, delivery_status: str,
                      detail: str = "",
                      last_fired_at: datetime | None = None,
                      ) -> Reminder | None:
        """Atomically update a contact reminder's delivery_status (and
        optionally last_fired_at). Returns the updated Reminder or None
        if not found. The `detail` arg is accepted for forward-compat /
        log-site clarity but is not persisted (no column today)."""
        def operation(session: Session) -> Reminder | None:
            reminder = session.get(Reminder, reminder_id)
            if reminder is None:
                return None
            reminder.delivery_status = delivery_status
            if last_fired_at is not None:
                reminder.last_fired_at = last_fired_at
                reminder.updated_at = last_fired_at
            else:
                reminder.updated_at = now_utc()
            session.flush()
            return self._coerce(reminder)
        return self._execute(operation)

    def start_delivery_attempt(
        self,
        reminder_id: str,
        *,
        attempt_started_at: datetime,
        prior_attempts_floor: int = 0,
    ) -> Reminder | None:
        def operation(session: Session) -> Reminder | None:
            reminder = session.get(Reminder, reminder_id)
            if reminder is None:
                return None
            current_attempts = int(getattr(reminder, 'delivery_attempts', 0) or 0)
            reminder.delivery_attempts = max(current_attempts, prior_attempts_floor) + 1
            reminder.delivery_status = 'requested'
            reminder.last_fired_at = attempt_started_at
            reminder.updated_at = attempt_started_at
            session.flush()
            return self._coerce(reminder)

        return self._execute(operation)

    def list_redelivery_candidates(
        self,
        *,
        fired_since: datetime,
    ) -> list[Reminder]:
        def operation(session: Session) -> list[Reminder]:
            stmt = (
                select(Reminder)
                .where(
                    Reminder.status == 'fired',
                    Reminder.last_fired_at.is_not(None),
                    Reminder.last_fired_at >= fired_since,
                    Reminder.delivery_status.in_(('failed', 'pending', 'requested')),
                )
                .order_by(Reminder.last_fired_at)
            )
            return [self._coerce(reminder) for reminder in session.scalars(stmt).all()]

        return self._execute(operation)

    def list_delivery_problems(self) -> list[Reminder]:
        """External-contact reminders whose delivery is stuck or gave up.

        Surfaces the delivery-tracking state for the dashboard: anything
        still 'requested'/'pending'/'failed' (in-flight or retrying) plus
        'abandoned' (exhausted the 3-attempt budget). Cancelled rows are
        excluded. Most-recently-fired first.
        """
        def operation(session: Session) -> list[Reminder]:
            stmt = (
                select(Reminder)
                .where(
                    Reminder.status != 'cancelled',
                    Reminder.delivery_status.in_(
                        ('requested', 'pending', 'failed', 'abandoned'),
                    ),
                )
                .order_by(Reminder.last_fired_at.desc())
            )
            return [self._coerce(reminder) for reminder in session.scalars(stmt).all()]

        return self._execute(operation)

    def claim_failure_alert(
        self,
        reminder_id: str,
        *,
        attempt_started_at: datetime,
        alerted_at: datetime,
    ) -> bool:
        def operation(session: Session) -> bool:
            stmt = (
                update(Reminder)
                .where(
                    Reminder.id == reminder_id,
                    Reminder.delivery_status == 'failed',
                    Reminder.last_fired_at == attempt_started_at,
                    or_(
                        Reminder.last_failure_alert_at.is_(None),
                        Reminder.last_failure_alert_at < Reminder.last_fired_at,
                    ),
                )
                .values(
                    last_failure_alert_at=alerted_at,
                    updated_at=alerted_at,
                )
            )
            result = session.execute(stmt)
            session.flush()
            return result.rowcount > 0

        return self._execute(operation)

    def should_fire(self, reminder_id: str) -> bool:
        now = now_utc()
        def operation(session: Session) -> bool:
            reminder = session.get(Reminder, reminder_id)
            if reminder is None or reminder.status != 'active':
                return False
            reminder = self._coerce(reminder)
            assert reminder is not None
            if reminder.last_fired_at is None:
                return reminder.next_fire_at <= now
            if reminder.recurrence is None:
                return reminder.last_fired_at < reminder.next_fire_at <= now
            return reminder.last_fired_at < reminder.next_fire_at <= now
        return self._execute(operation)
