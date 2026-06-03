"""Destructive intent classifier — H2-039 FIX 1 Pattern B.

Brain_router → claude -p subprocess model means the bot never sees individual
MCP tool calls; Claude executes the entire tool loop internally. So per-tool
approval enforcement isn't reachable from the bot side.

Pattern B trades that for **per-message** approval: classify the user's prompt
for destructive intent BEFORE handing off to Claude. If destructive, surface
an ApprovalService prompt; only invoke Claude after the user taps approve.

This module is pure — no Telegram, no DB, no I/O. Token-set matching:
each tool registers (action_words, target_words). A tool fires when at
least one action AND at least one target token are both present in the
prompt (in any order, anywhere). The tool whose match is most specific
(matched both a strong action AND a specific target) wins.

The MCP-server-side @mcp.tool(meta={"approval_required": True, ...}) is
documentation that mirrors this registry — Claude CLI in -p mode doesn't
enforce it, so this classifier is the operational gate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from services.prompt_content_filters import strip_embedded_content


@dataclass(frozen=True, slots=True)
class DestructiveIntent:
    is_destructive: bool
    confidence: float
    matched_tools: tuple[str, ...] = ()
    matched_verbs: tuple[str, ...] = ()
    suggested_approval_template: str = ""
    # H2-047 Fix 2: when True, the dispatcher must NOT arm an approval prompt
    # (no Approve/Cancel buttons). The classifier wants to refuse / advise
    # rather than gate. Currently set by the multi-intent path which used to
    # confusingly show "Multiple actions detected" with Approve/Cancel buttons
    # — the buttons offered "proceed" while the text said "send one at a time",
    # so users tapped Approve out of habit and bypassed the very check we
    # asked them to make.
    is_advisory_only: bool = False


# ---------- Token-set registry ----------
# tool_name -> (action_tokens_any, target_tokens_any, approval_template)
#
# A tool matches when the prompt contains at least one action token AND
# at least one target token. Empty target list means "action alone matches"
# (used for tools like send_email where "send email" is the natural phrasing).
#
# Tokens are whole-word matched (case-insensitive). Multi-word tokens like
# "fill out" are allowed and use \s+ between words. The order in the
# prompt doesn't matter.

DESTRUCTIVE_TOOL_REGISTRY: dict[str, tuple[tuple[str, ...], tuple[str, ...], str]] = {
    # ── nexus-filesystem ──
    "write_text_file":     (("write", "save", "create"),     ("file", "txt", "log"),       "Write to a file?"),
    "append_to_file":      (("append", "add"),                ("file", "log"),              "Append to a file?"),
    "move_file":           (("move", "rename"),               ("file",),                    "Move/rename a file?"),
    "delete_file":         (("delete", "remove", "rm"),       ("file",),                    "Delete a file?"),

    # ── nexus-pdf-docs ──
    "create_pdf_from_text":     (("create", "make", "generate"), ("pdf",),                  "Generate a PDF?"),
    "create_pdf_from_template": (("fill", "use", "render"),       ("template",),             "Generate a PDF from template?"),
    "fill_pdf_form":            (("fill",),                       ("form", "pdf"),           "Fill out a PDF form?"),
    "merge_pdfs":               (("merge", "combine"),            ("pdf", "pdfs"),           "Merge PDFs?"),
    "split_pdf":                (("split", "extract"),            ("pdf", "pages"),          "Split a PDF?"),
    "add_signature_to_pdf":     (("sign", "signature"),           ("pdf", "contract", "document"), "Add signature to a PDF?"),

    # ── nexus-email ──
    "send_email":         (("send",),                            ("email", "mail"),                  "Send an email?"),
    "forward_email":      (("forward",),                          ("email", "message", "thread", "invoice", "newsletter", "mail"),    "Forward an email?"),
    "reply_to_email":     (("reply",),                            ("email", "message", "thread"),    "Reply to an email?"),
    "archive_email":      (("archive",),                          ("email", "message", "thread"),    "Archive an email?"),
    "delete_email":       (("delete", "trash"),                   ("email", "message", "inbox"),     "Delete an email?"),
    "extract_attachments":(("extract", "save", "download"),       ("attachment", "attachments"),     "Save attachments to disk?"),

    # ── nexus-calendar ──
    "create_event":       (("schedule", "create", "book", "add"),  ("event", "meeting", "appointment", "calendar"), "Create a calendar event?"),
    "update_event":       (("update", "reschedule", "change"),     ("event", "meeting"),              "Update a calendar event?"),
    "cancel_event":       (("cancel",),                            ("meeting", "event", "appointment"), "Cancel a calendar event?"),
    "accept_invite":      (("accept",),                            ("invite", "invitation"),          "Accept a calendar invite?"),
    "decline_invite":     (("decline", "reject"),                  ("invite", "invitation"),          "Decline a calendar invite?"),
    "tentative_invite":   (("tentative", "maybe"),                 ("invite", "invitation"),          "Mark invite tentative?"),
    "block_focus_time":   (("block",),                             ("focus", "time"),                 "Block focus time on calendar?"),

    # ── nexus-contacts ──
    "create_contact":     (("add", "create", "new"),               ("contact",),                       "Create a contact?"),
    "update_contact":     (("update", "edit", "change"),           ("contact",),                       "Update a contact?"),
    "delete_contact":     (("delete", "remove"),                   ("contact",),                       "Delete a contact?"),
    "add_to_group":       (("add",),                               ("group",),                         "Add contact to a group?"),
    "remove_from_group":  (("remove",),                             ("group",),                        "Remove contact from a group?"),
    "import_vcard":       (("import",),                            ("vcard", "contacts"),              "Import vCard contacts?"),

    # ── nexus-reminders-tasks ──
    # Reminder edit/cancel are intentionally NOT registered (owner directive
    # 2026-06-02: "No approval gates on reminders"). Acting on the user's own
    # reminder is low-risk and the approval prompt made it feel broken. The
    # MCP-server meta flags were removed to match. delete_task stays gated.
    "delete_task":        (("delete", "remove"),                    ("task",),                         "Delete a task?"),

    # ── nexus-messaging ──
    "send_message":       (("send", "text", "whatsapp"),            ("message", "msg", "whatsapp"),    "Send a message? (double-confirm)"),
    "send_official_notice": (("send",),                             ("lease renewal", "late fee notice", "legal notice"), "Send an official notice?"),
    "assign_fault":       (("tell", "notify", "inform"),            ("responsible", "at fault"),       "Send a liability or fault decision?"),
    "share_tenant_information": (("forward", "share", "send"),      ("tenant info", "tenant phone", "tenant contact", "vendor"), "Share tenant information with a third party?"),

    # ── nexus-web ──
    "book_appointment":   (("book", "schedule", "reserve"),         ("appointment", "haircut", "slot", "reservation"), "Book an appointment?"),
    "fill_web_form":      (("fill", "submit"),                      ("form",),                         "Fill out and submit a web form?"),

    # ── nexus-rentals ──
    "delete_unit":        (("delete", "remove"),                    ("unit", "rental"),                "Delete a rental unit?"),
    # H2-048: update_unit MCP tool shipped in H2-047 but its approval
    # template was buried under the generic-verb fallback because there
    # was no registry entry; "update unit 1 lease_end" now surfaces the
    # specific "Update rental unit?" prompt instead of the meaningless
    # "This may modify or send data" generic.
    "update_unit":        (("update", "change", "modify", "edit", "mark"), ("unit", "rental", "lease"), "Update rental unit?"),
    "add_payment":        (("record", "log", "add", "pay"),         ("payment", "rent"),               "Record a rental payment?"),
    "add_maintenance":    (("log", "report", "file"),                ("maintenance", "issue", "ticket"),  "Log a maintenance request?"),
    "approve_vendor_action": (("approve", "authorize", "accept"),   ("quote", "estimate", "vendor", "repair", "handyman"), "Approve vendor or repair work?"),
    "close_case":         (("close", "dismiss", "mark"),             ("case", "issue", "resolved", "maintenance"), "Close or resolve a case or issue?"),
    "update_payment_status": (("pay", "charge", "mark"),            ("invoice", "late fee", "rent paid", "rent unpaid"), "Change payment or rent status?"),

    # ── nexus-business ──
    "delete_business_item": (("delete", "remove"),                   ("business",),                    "Delete a business item?"),
    "delete_llc":           (("delete", "remove", "dissolve"),       ("llc",),                         "Delete an LLC?"),
    "delete_quick_link":    (("delete", "remove"),                   ("link",),                        "Delete a quick link?"),

    # ── nexus-knowledge ──
    "forget":               (("forget", "wipe"),                     (),                               "Forget a memory entry?"),
    "update_memory":        (("update", "edit", "rewrite"),          ("memory",),                      "Update a memory entry?"),

    # ── nexus-utils (notes + cron) ──
    "delete_note":          (("delete", "remove"),                   ("note",),                        "Delete a note?"),
    "update_note":          (("update", "edit", "rewrite"),          ("note",),                        "Update a note?"),
    "delete_cron_job":      (("delete", "remove"),                   ("cron", "job"),                  "Delete a cron job?"),
    "create_cron_job":      (("schedule", "create", "add"),          ("cron", "recurring"),            "Create a recurring cron job?"),
}


# Generic destructive verbs — fire even when no specific tool matches.
_GENERIC_DESTRUCTIVE_VERBS: tuple[str, ...] = (
    "delete", "remove", "destroy", "drop", "wipe", "purge",
    "send", "forward", "reply", "compose", "dispatch",
    "cancel", "archive",
    "schedule", "book", "reserve",
    "move", "rename",
    "write", "append", "overwrite",
    "fill out", "submit",
    "sign",
    "record payment", "pay",
    "forget",
    "update", "edit", "change", "modify",
)

# Read-only verbs — heavy presence downgrades a generic-verb hit.
# Single-word "what" is intentionally excluded — common in destructive
# prompts ("forget what I told you") and the multi-word forms catch the
# real questions.
_READ_VERBS: tuple[str, ...] = (
    "show", "list", "what's", "whats", "what is",
    "find", "search", "look up", "lookup",
    "summary", "summarize", "tell me", "how many", "do i have",
    "view", "check", "status",
    "when is", "where is", "who is",
)


# ---------- Low-risk fast path (2026-05-31) ----------
# NEXUS now has enough safety; the next evolution is judgment about when NOT
# to gate. Questions / advice / analysis / opinions / document review /
# brainstorming / planning are the user asking the assistant to *think*, not
# to *act* — they must never trip an approval prompt, even when they mention
# a destructive verb ("should I update my resume?", "what's your opinion on
# whether to send the notice?"). Only an actual operation
# (sending/deleting/modifying/scheduling/writing) enters approval mode.
#
# An advisory prompt is one framed as a request to think. When that framing
# is present AND the prompt is not also an explicit go-ahead command, the
# classifier returns non-destructive before any verb matching runs.

_ADVISORY_MARKERS: tuple[str, ...] = (
    # opinions / questions
    "what do you think", "what's your take", "whats your take", "your take on",
    "your opinion", "your thoughts", "any thoughts", "thoughts on",
    "do you think", "what do you make of", "how do you feel about", "wdyt",
    "what do you reckon",
    # advice
    "should i", "is it worth", "is it a good idea", "good idea to",
    "what would you do", "how would you", "what would you", "how should i",
    "what do you recommend", "do you recommend", "any advice", "advice on",
    "advise me", "what do you suggest", "any suggestions", "what do you advise",
    # analysis / comparison
    "pros and cons", "pros or cons", "which is better", "what's better",
    "whats better", "what are my options", "help me decide", "compare",
    "can you analyze", "analyze this", "analysis of", "evaluate", "assess",
    "weigh in", "what's the difference", "whats the difference",
    # document review
    "review this", "review the", "look over", "take a look", "look at this",
    "go over", "feedback on", "give me feedback", "critique", "proofread",
    # explanation / understanding
    "explain", "help me understand", "what does this mean", "what does it mean",
    "break down", "walk me through", "summarize", "tldr", "tl;dr",
    # brainstorming
    "brainstorm", "help me think", "let's think", "lets think",
    "ideas for", "come up with ideas", "what are some ideas",
    # planning
    "help me plan", "let's plan", "lets plan", "think through", "talk through",
    "plan for", "how do i approach", "what's a good plan", "whats a good plan",
)

# Explicit go-ahead / imperative markers that OVERRIDE advisory framing so a
# real command buried in an advisory-looking message still gates. Kept tight:
# advice questions ("should I delete this?") must NOT match — only clear
# execution intent does.
_EXPLICIT_COMMAND_MARKERS: tuple[str, ...] = (
    "do it", "just do it", "go ahead", "go for it", "make it happen",
    "get it done", "handle it", "make it so", "do this now", "right now",
    "asap", "immediately", "send it now", "delete it now",
)

# Compound "…and then send/delete/…" — an advisory clause followed by an
# imperative action verb is a real command, not a question.
_COMPOUND_COMMAND_RE = re.compile(
    r"\b(?:and|then|also|,)\s+(?:please\s+)?(?:"
    r"send|delete|remove|cancel|forward|reply|archive|schedule|book|reserve|"
    r"write|append|overwrite|move|rename|sign|pay|submit|fill out|dispatch"
    r")\b"
)

# "just send/delete/…" — emphatic imperative.
_JUST_COMMAND_RE = re.compile(
    r"\bjust\s+(?:please\s+)?(?:"
    r"send|delete|remove|cancel|forward|archive|schedule|book|"
    r"write|move|rename|sign|pay|submit|do)\b"
)


def _has_explicit_command(text: str) -> bool:
    if any(_token_present(text, m) for m in _EXPLICIT_COMMAND_MARKERS):
        return True
    return bool(_COMPOUND_COMMAND_RE.search(text) or _JUST_COMMAND_RE.search(text))


def _is_advisory_request(text: str) -> bool:
    """True when the prompt asks the assistant to think/opine/analyze/review/
    brainstorm/plan, rather than to perform an operation."""
    return any(_token_present(text, marker) for marker in _ADVISORY_MARKERS)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _token_present(text: str, token: str) -> bool:
    # Multi-word tokens use \s+ to tolerate extra spaces.
    pattern = r"\b" + r"\s+".join(re.escape(w) for w in token.split()) + r"\b"
    return bool(re.search(pattern, text))


def _any_present(text: str, tokens: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(t for t in tokens if _token_present(text, t))


_MULTI_INTENT_NUMBERED_RE = re.compile(
    # Numbered list items at line start or after whitespace.
    # Pattern: a digit followed by . or ) and then a verb/word.
    # Multi-line OR same-line separated by spaces — both common in
    # the production "1. Send the 3 leases 2. Do it 3. I want to wire..."
    # pattern that arrives from Telegram as a single message.
    r"(?:^|\s)(\d+)[\.)]\s+\S",
    flags=re.MULTILINE,
)


def _is_multi_intent(text: str) -> bool:
    """Detect numbered-list multi-action prompts.

    Fires when the (already lowercased + whitespace-collapsed) text shows
    ≥2 distinct numbered list markers like ``1. ... 2. ...``. The classifier
    cannot reliably gate such prompts as a single approval because each
    numbered item may be a distinct destructive action; better to refuse
    to bundle and ask the user to break them up.
    """
    matches = list(_MULTI_INTENT_NUMBERED_RE.finditer(text))
    if len(matches) < 2:
        return False
    # Require the numbers themselves to be distinct and increasing (1, 2, 3...)
    # so a phrase like "I have 1 thing here and 2 things there" — which has
    # two digit-then-period-ish hits — doesn't false-positive.
    numbers = []
    for m in matches:
        try:
            numbers.append(int(m.group(1)))
        except (ValueError, TypeError):
            continue
    if len(numbers) < 2:
        return False
    distinct = sorted(set(numbers))
    return len(distinct) >= 2


def classify(prompt: str) -> DestructiveIntent:
    """Classify a user prompt for destructive intent.

    Token-set match: a tool fires when prompt contains ≥1 action token AND
    (≥1 target token OR empty target list). When multiple tools match, the
    one with the most specific composite hit (action_hits + target_hits)
    wins for the suggested template; all matched tools are returned.

    Conservative bias: ambiguous → destructive.

    H2-046:
      - Strip embedded document blocks before matching so PDF/transcript
        body text doesn't trip per-tool keyword rules.
      - Detect numbered multi-intent prompts before per-tool matching and
        refuse to bundle them under a single approval template.
    """
    # H2-046: classification operates on the user's *instruction*, not on
    # embedded document body text. PDFs and transcripts arrive between
    # known delimiters; strip them before any matching.
    prompt = strip_embedded_content(prompt)
    text = _normalize(prompt)
    if not text:
        return DestructiveIntent(is_destructive=False, confidence=0.0)

    # Approval-callback strings always bypass.
    if text.startswith("approval:approve:") or text.startswith("approval:cancel:"):
        return DestructiveIntent(is_destructive=False, confidence=0.0)

    # H2-046: multi-intent numbered prompts collapse to one approval today.
    # Refuse to bundle; ask the user to break them up instead.
    # H2-047 Fix 2: emit as advisory text (no Approve/Cancel buttons). The
    # dispatcher renders this as a plain reply.
    if _is_multi_intent(text):
        return DestructiveIntent(
            is_destructive=True,
            confidence=0.7,
            matched_tools=(),
            matched_verbs=("multi_intent",),
            suggested_approval_template=(
                "Multiple actions detected — please send them one at a time."
            ),
            is_advisory_only=True,
        )

    # Low-risk fast path: a question / advice / analysis / opinion / document
    # review / brainstorm / planning request asks the assistant to *think*,
    # not to *act*. Never gate it — even if it mentions a destructive verb —
    # unless it also carries an explicit go-ahead command ("…and send it",
    # "just delete it", "do it now"). Operations fall through to gating below.
    if _is_advisory_request(text) and not _has_explicit_command(text):
        return DestructiveIntent(
            is_destructive=False,
            confidence=0.0,
            matched_verbs=("advisory_fast_path",),
        )

    # Per-tool token-set match.
    tool_matches: list[tuple[int, str, tuple[str, ...], str]] = []
    # (specificity, tool_name, hit_verbs, template)

    for tool_name, (actions, targets, template) in DESTRUCTIVE_TOOL_REGISTRY.items():
        action_hits = _any_present(text, actions)
        if not action_hits:
            continue
        if targets:
            target_hits = _any_present(text, targets)
            if not target_hits:
                continue
            specificity = len(action_hits) + len(target_hits)
            verbs = action_hits + target_hits
        else:
            specificity = len(action_hits)
            verbs = action_hits
        tool_matches.append((specificity, tool_name, verbs, template))

    if tool_matches:
        # Read-verb downgrade: if the prompt is dominated by read-style verbs
        # ("how many", "show me", "what is"), the tool match is incidental —
        # the user is asking ABOUT a destructive thing, not requesting one.
        read_hits = _any_present(text, _READ_VERBS)
        action_token_count = sum(len(m[2]) for m in tool_matches)
        if len(read_hits) >= action_token_count and len(read_hits) >= 1:
            return DestructiveIntent(
                is_destructive=False, confidence=0.3,
                matched_verbs=tuple(sorted(set(read_hits))),
            )
        # Sort by specificity desc, take winner for template.
        tool_matches.sort(key=lambda x: (-x[0], x[1]))
        winner_template = tool_matches[0][3]
        all_tools = tuple(sorted({m[1] for m in tool_matches}))
        all_verbs = tuple(sorted({v for m in tool_matches for v in m[2]}))
        return DestructiveIntent(
            is_destructive=True, confidence=0.85,
            matched_tools=all_tools, matched_verbs=all_verbs,
            suggested_approval_template=winner_template,
        )

    # Reminder fast path (owner directive 2026-06-02: "No approval gates on
    # reminders"). By this point NO concrete destructive tool matched — a
    # compound prompt that also targets a real destructive tool ("delete the
    # reminder AND the file") already returned above via tool_matches and still
    # gates. So if what's left is a bare reminder operation, act immediately:
    # editing/cancelling the user's own reminder is low-risk, and the approval
    # prompt is exactly what made reminders feel broken.
    if _token_present(text, "reminder") or _token_present(text, "reminders"):
        return DestructiveIntent(
            is_destructive=False, confidence=0.0,
            matched_verbs=("reminder_fast_path",),
        )

    # Generic-verb match: medium confidence.
    generic_hits = _any_present(text, _GENERIC_DESTRUCTIVE_VERBS)
    if generic_hits:
        read_hits = _any_present(text, _READ_VERBS)
        if len(read_hits) > len(generic_hits):
            return DestructiveIntent(
                is_destructive=False, confidence=0.4,
                matched_verbs=tuple(sorted(set(generic_hits))),
            )
        return DestructiveIntent(
            is_destructive=True, confidence=0.6,
            matched_verbs=tuple(sorted(set(generic_hits))),
            suggested_approval_template="This may modify or send data. Proceed?",
        )

    return DestructiveIntent(is_destructive=False, confidence=0.0)


def render_default_preview(prompt: str, intent: DestructiveIntent) -> str:
    """Compose a Telegram-ready preview string for ApprovalService.request."""
    cap = 240
    snippet = prompt.strip()
    if len(snippet) > cap:
        snippet = snippet[: cap - 1].rstrip() + "…"
    template = intent.suggested_approval_template or "This may modify or send data. Proceed?"
    return f"{template}\n\nYou asked: “{snippet}”"
