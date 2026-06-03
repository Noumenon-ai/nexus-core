from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import Boolean, CheckConstraint, Date, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship



def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base ORM model."""


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(default=now_utc, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(default=now_utc, onupdate=now_utc, nullable=False)


class User(Base, TimestampMixin):
    __tablename__ = 'users'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    telegram_id: Mapped[int] = mapped_column(unique=True, nullable=False, index=True)
    username: Mapped[str | None] = mapped_column(String(255))
    full_name: Mapped[str | None] = mapped_column(String(255))
    language: Mapped[str] = mapped_column(String(8), default='en', nullable=False)
    role: Mapped[str] = mapped_column(String(16), default='user', nullable=False)
    proactive_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    voice_output_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    elevenlabs_voice_id: Mapped[str | None] = mapped_column(Text)
    elevenlabs_api_key: Mapped[str | None] = mapped_column(Text)
    elevenlabs_voice_settings: Mapped[str | None] = mapped_column(Text)
    elevenlabs_migrated_at: Mapped[datetime | None]
    google_connected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    google_email: Mapped[str | None] = mapped_column(Text)
    google_calendar_primary_id: Mapped[str | None] = mapped_column(Text)
    google_scopes_granted: Mapped[str | None] = mapped_column(Text)
    google_connected_at: Mapped[datetime | None]

    reminders: Mapped[list['Reminder']] = relationship(back_populates='user')
    tasks: Mapped[list['Task']] = relationship(back_populates='user')
    memories: Mapped[list['Memory']] = relationship(back_populates='user')
    contexts: Mapped[list['ConversationContext']] = relationship(back_populates='user')
    emails: Mapped[list['EmailIngested']] = relationship(back_populates='user')
    approvals: Mapped[list['Approval']] = relationship(back_populates='user')
    proactive_notifications: Mapped[list['ProactiveNotification']] = relationship(back_populates='user')
    opportunity_signals: Mapped[list['OpportunitySignal']] = relationship(back_populates='user')
    elevenlabs_usage_records: Mapped[list['ElevenLabsUsage']] = relationship(back_populates='user')


class Reminder(Base, TimestampMixin):
    __tablename__ = 'reminders'
    __table_args__ = (Index('idx_reminders_active', 'status', 'next_fire_at'),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), nullable=False, index=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    next_fire_at: Mapped[datetime] = mapped_column(nullable=False)
    recurrence: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    last_fired_at: Mapped[datetime | None]
    # Delivery reliability for normal (Telegram) reminders: redelivery sweep
    # and boot recovery use these to retry a fire that failed to send.
    delivery_status: Mapped[str | None] = mapped_column(String(16),
                                                        nullable=True)
    delivery_attempts: Mapped[int] = mapped_column(Integer, default=0,
                                                   nullable=False,
                                                   server_default=text('0'))
    last_failure_alert_at: Mapped[datetime | None]

    user: Mapped[User] = relationship(back_populates='reminders')


class Task(Base, TimestampMixin):
    __tablename__ = 'tasks'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    due_at: Mapped[datetime | None]
    priority: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    source: Mapped[str | None] = mapped_column(String(32))
    completed_at: Mapped[datetime | None]

    user: Mapped[User] = relationship(back_populates='tasks')


class Memory(Base, TimestampMixin):
    __tablename__ = 'memories'
    __table_args__ = (UniqueConstraint('user_id', 'memory_type', 'key', name='uq_memory_user_type_key'),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), nullable=False, index=True)
    memory_type: Mapped[str] = mapped_column(String(32), nullable=False)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)

    user: Mapped[User] = relationship(back_populates='memories')


class ConversationContext(Base, TimestampMixin):
    __tablename__ = 'conversation_context'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), nullable=False, index=True)
    last_topic: Mapped[str | None] = mapped_column(Text)
    last_entity_type: Mapped[str | None] = mapped_column(String(64))
    last_entity_value: Mapped[str | None] = mapped_column(Text)
    last_intent: Mapped[str | None] = mapped_column(String(64))
    context_json: Mapped[str | None] = mapped_column(Text)
    expires_at: Mapped[datetime] = mapped_column(nullable=False)

    user: Mapped[User] = relationship(back_populates='contexts')


class EmailIngested(Base):
    __tablename__ = 'emails_ingested'
    __table_args__ = (UniqueConstraint('user_id', 'gmail_message_id', name='uq_email_user_message'),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), nullable=False, index=True)
    gmail_message_id: Mapped[str] = mapped_column(String(255), nullable=False)
    subject: Mapped[str | None] = mapped_column(Text)
    sender: Mapped[str | None] = mapped_column(Text)
    snippet: Mapped[str | None] = mapped_column(Text)
    received_at: Mapped[datetime] = mapped_column(nullable=False)
    category: Mapped[str | None] = mapped_column(String(32))
    extracted_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=now_utc, nullable=False)

    user: Mapped[User] = relationship(back_populates='emails')


class Approval(Base):
    __tablename__ = 'approvals'
    __table_args__ = (Index('idx_approvals_pending', 'status', 'expires_at'),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), nullable=False, index=True)
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    preview_text: Mapped[str] = mapped_column(Text, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=now_utc, nullable=False)
    approved_at: Mapped[datetime | None]
    cancelled_at: Mapped[datetime | None]
    executed_at: Mapped[datetime | None]

    user: Mapped[User] = relationship(back_populates='approvals')


class AuditEvent(Base):
    """Delivery-truth audit row. One per executed (or attempted) action.

    Step 5 of NEXUS_ARCHITECTURE_REFACTOR.md: "If it's not in
    audit_events — it didn't happen." Every IntentExecutor branch
    (capability-blocked, approval-deferred, safety-blocked, executed)
    writes one row. The reasoning_adapter, dispatcher direct paths,
    and approvals all funnel here.
    """
    __tablename__ = 'audit_events'
    __table_args__ = (
        Index('idx_audit_events_user_ts', 'user_id', 'created_at'),
        Index('idx_audit_events_intent', 'intent'),
        Index('idx_audit_events_result', 'result'),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    created_at: Mapped[datetime] = mapped_column(
        default=now_utc, nullable=False, index=True,
    )
    user_id: Mapped[str | None] = mapped_column(
        ForeignKey('users.id'), nullable=True, index=True,
    )
    intent: Mapped[str] = mapped_column(String(64), nullable=False)
    parameters_json: Mapped[str] = mapped_column(Text, nullable=False, default='{}')
    result: Mapped[str] = mapped_column(String(64), nullable=False)
    response_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)


class ProactiveNotification(Base):
    __tablename__ = 'proactive_notifications'
    __table_args__ = (Index('idx_proactive_dedupe', 'user_id', 'dedupe_key', 'last_sent_at'),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), nullable=False, index=True)
    notification_type: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    dedupe_key: Mapped[str] = mapped_column(String(255), nullable=False)
    last_sent_at: Mapped[datetime] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=now_utc, nullable=False)

    user: Mapped[User] = relationship(back_populates='proactive_notifications')


class OpportunitySignal(Base):
    __tablename__ = 'opportunity_signals'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    dedupe_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=now_utc, nullable=False)

    user: Mapped[User] = relationship(back_populates='opportunity_signals')


class ElevenLabsUsage(Base):
    __tablename__ = 'elevenlabs_usage'
    __table_args__ = (UniqueConstraint('user_id', 'day', name='uq_elevenlabs_usage_user_day'),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str | None] = mapped_column(ForeignKey('users.id'), index=True)
    day: Mapped[date] = mapped_column(Date, nullable=False)
    chars_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    api_calls: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    user: Mapped[User | None] = relationship(back_populates='elevenlabs_usage_records')


class ConversationTurn(Base):
    """V3.6 forward-only conversation archive.

    Every dispatcher handle() writes one user turn (on entry) and one
    assistant turn (after final reply composed). mem0 persistence is
    fire-and-forget background task that updates mem0_persisted_at +
    mem0_memory_id on success. Rows with mem0_persisted_at IS NULL
    after a grace window are recoverable via the partial index.

    Pre-V3.5 conversation history is NOT in this table — see
    HARDENING_PASS_V2.md H2-020 for the unrecoverable-history finding.
    """
    __tablename__ = 'conversation_turns'

    turn_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=now_utc, nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(36), nullable=False)
    mem0_persisted_at: Mapped[datetime | None] = mapped_column()
    mem0_memory_id: Mapped[str | None] = mapped_column(String(255))

    __table_args__ = (
        Index('idx_conversation_turns_user_created', 'user_id', 'created_at'),
        Index('idx_conversation_turns_conv', 'conversation_id'),
        Index(
            'idx_conversation_turns_mem0_pending',
            'user_id', 'created_at',
            sqlite_where=text('mem0_persisted_at IS NULL'),
        ),
    )


class TelosOnboardingState(Base):
    """V3.8 TELOS onboarding flow state.

    One row per user (user_id PRIMARY KEY). Tracks which section the
    user is currently answering, the JSON-serialized answers gathered
    so far, and the lifecycle timestamps. `started_at` is nullable
    (spec deviation from V3.8 audit, 2026-05-04): a row can exist
    BEFORE onboarding actually starts because the briefing-nudge
    bookkeeping (`last_nudge_at`) writes to this same table — that
    write happens whenever the nudge fires, even for a user who never
    runs `start_telos_onboarding`. The first start sets `started_at`.

    `cancelled_at` lets `cancel_telos_onboarding` mark a pause without
    losing `current_section` or `answers_so_far`, so resume picks up
    at the right question (V3.8 halt condition: cancel must preserve
    state).
    """
    __tablename__ = 'telos_onboarding_state'

    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), primary_key=True)
    current_section: Mapped[str] = mapped_column(String(64), default='identity', nullable=False)
    answers_so_far: Mapped[str] = mapped_column(Text, default='{}', nullable=False)
    started_at: Mapped[datetime | None] = mapped_column()
    completed_at: Mapped[datetime | None] = mapped_column()
    cancelled_at: Mapped[datetime | None] = mapped_column()
    last_nudge_at: Mapped[datetime | None] = mapped_column()


class CronJob(Base, TimestampMixin):
    """Phase 6: user-defined cron jobs.

    APScheduler runs these (cron_expression compiled to CronTrigger).  When
    fired, the action_text is injected into the bot pipeline as if the user
    just typed it — see scheduler.dispatch_cron_job.

    `pending_sync` is `1` when the row needs APScheduler to re-read it (new,
    modified, paused, resumed).  The bot's sync loop clears it back to `0`
    after applying changes.
    """
    __tablename__ = 'cron_jobs'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    cron_expression: Mapped[str] = mapped_column(String(128), nullable=False)
    action_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default='active', nullable=False)  # active|paused|deleted
    pending_sync: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    last_fired_at: Mapped[datetime | None] = mapped_column()
    fire_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


