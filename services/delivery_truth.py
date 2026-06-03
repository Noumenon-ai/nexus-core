from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


DELIVERY_REQUESTED = "requested"
DELIVERY_APPROVED = "approved"
DELIVERY_QUEUED = "queued"
DELIVERY_SENT_TO_SERVER = "sent_to_server"
DELIVERY_DELIVERED = "delivered"
DELIVERY_READ = "read"
DELIVERY_FAILED = "failed"
DELIVERY_UNKNOWN = "unknown"

# Backward-compat for rows written before Phase 6.
LEGACY_PENDING = "pending"
LEGACY_SENT = "sent"

RETRY_IN_PROGRESS_STATES = frozenset({DELIVERY_REQUESTED, LEGACY_PENDING})
DISPATCHED_DELIVERY_STATES = frozenset(
    {
        DELIVERY_REQUESTED,
        DELIVERY_SENT_TO_SERVER,
        DELIVERY_DELIVERED,
        DELIVERY_READ,
        DELIVERY_UNKNOWN,
        LEGACY_PENDING,
        LEGACY_SENT,
    }
)


@dataclass(frozen=True, slots=True)
class DeliveryTruth:
    state: str
    user_text: str
    ack_status: int | None = None
    ack_text: str | None = None
    message_id: str | None = None
    platform: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def normalize_dispatch_delivery(result: Mapping[str, Any] | None) -> DeliveryTruth:
    payload = dict(result or {})
    platform = str(payload.get("platform") or "").strip().lower() or None
    message_id = _clean_str(payload.get("message_id"))
    if payload.get("ok") is not True:
        return DeliveryTruth(
            state=DELIVERY_FAILED,
            user_text="Failed.",
            message_id=message_id,
            platform=platform,
        )
    if platform == "whatsapp":
        return normalize_whatsapp_delivery(
            payload.get("delivery"),
            message_id=message_id,
        )
    if platform == "sms":
        return DeliveryTruth(
            state=DELIVERY_UNKNOWN,
            user_text="The SMS provider accepted the request, but delivery status is unknown.",
            message_id=message_id,
            platform=platform,
        )
    return DeliveryTruth(
        state=DELIVERY_UNKNOWN,
        user_text="The provider accepted the request, but delivery status is unknown.",
        message_id=message_id,
        platform=platform,
    )


def normalize_whatsapp_delivery(
    delivery: Mapping[str, Any] | None,
    *,
    message_id: str | None = None,
) -> DeliveryTruth:
    payload = dict(delivery or {})
    status = _coerce_int(payload.get("status"))
    ack_text = _clean_str(payload.get("status_text"))

    if status is not None:
        if status <= 0:
            return DeliveryTruth(
                state=DELIVERY_FAILED,
                user_text="WhatsApp reported a send failure.",
                ack_status=status,
                ack_text=ack_text or "error",
                message_id=message_id,
                platform="whatsapp",
            )
        if status == 1:
            return DeliveryTruth(
                state=DELIVERY_QUEUED,
                user_text="Queued.",
                ack_status=status,
                ack_text=ack_text or "pending",
                message_id=message_id,
                platform="whatsapp",
            )
        if status == 2:
            return DeliveryTruth(
                state=DELIVERY_SENT_TO_SERVER,
                user_text="WhatsApp accepted it. Delivery is still pending.",
                ack_status=status,
                ack_text=ack_text or "server_ack",
                message_id=message_id,
                platform="whatsapp",
            )
        if status == 3:
            return DeliveryTruth(
                state=DELIVERY_DELIVERED,
                user_text="Delivered.",
                ack_status=status,
                ack_text=ack_text or "delivery_ack",
                message_id=message_id,
                platform="whatsapp",
            )
        if status >= 4:
            return DeliveryTruth(
                state=DELIVERY_READ,
                user_text="Read.",
                ack_status=status,
                ack_text=ack_text or "read",
                message_id=message_id,
                platform="whatsapp",
            )

    lowered = (ack_text or "").casefold()
    if lowered in {"error", "failed", "send_failed"}:
        return DeliveryTruth(
            state=DELIVERY_FAILED,
            user_text="WhatsApp reported a send failure.",
            ack_text=ack_text,
            message_id=message_id,
            platform="whatsapp",
        )
    if lowered in {"pending", "queued"}:
        return DeliveryTruth(
            state=DELIVERY_QUEUED,
            user_text="Queued.",
            ack_text=ack_text,
            message_id=message_id,
            platform="whatsapp",
        )
    if lowered in {"server_ack", "accepted", "sent_to_server"}:
        return DeliveryTruth(
            state=DELIVERY_SENT_TO_SERVER,
            user_text="WhatsApp accepted it. Delivery is still pending.",
            ack_text=ack_text,
            message_id=message_id,
            platform="whatsapp",
        )
    if lowered in {"delivery_ack", "delivered"}:
        return DeliveryTruth(
            state=DELIVERY_DELIVERED,
            user_text="Delivered.",
            ack_text=ack_text,
            message_id=message_id,
            platform="whatsapp",
        )
    if lowered in {"read", "played"}:
        return DeliveryTruth(
            state=DELIVERY_READ,
            user_text="Read.",
            ack_text=ack_text,
            message_id=message_id,
            platform="whatsapp",
        )

    return DeliveryTruth(
        state=DELIVERY_UNKNOWN,
        user_text="I sent it to WhatsApp, but delivery status is unknown.",
        ack_text=ack_text,
        message_id=message_id,
        platform="whatsapp",
    )


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
