"""
Support inbox copilot: drafts replies, flags urgency, maintains audit log.
Human-in-the-loop: drafts are reviewed before sending.
"""
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from classifier import (SupportEmailClassifier, EmailMessage,
                             ClassificationResult, UrgencyLevel, SupportIntent)
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False


class DraftStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"
    SENT = "sent"


@dataclass
class ReplyDraft:
    draft_id: str
    email_id: str
    subject: str
    body: str
    classification: Optional[ClassificationResult]
    status: DraftStatus = DraftStatus.PENDING_REVIEW
    generated_at: float = field(default_factory=time.time)
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[float] = None
    final_body: Optional[str] = None


@dataclass
class CopilotAuditEntry:
    entry_id: str
    email_id: str
    action: str
    actor: str
    details: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class ReplyTemplateLibrary:
    """Pre-built reply templates for common intents."""

    TEMPLATES: Dict[str, str] = {
        "order_status": (
            "Dear {customer_name},\n\n"
            "Thank you for reaching out. We have looked up your order {order_id} "
            "and can confirm that it is currently {status}. You can track your "
            "shipment using this link: {tracking_url}.\n\n"
            "If you have any further questions, please do not hesitate to contact us.\n\n"
            "Best regards,\nSupport Team"
        ),
        "refund_request": (
            "Dear {customer_name},\n\n"
            "We have received your refund request regarding order {order_id}. "
            "Our team will review your request within 2-3 business days. "
            "Once approved, the refund will be processed to your original payment method "
            "within 5-7 business days.\n\n"
            "We apologize for the inconvenience caused.\n\n"
            "Best regards,\nSupport Team"
        ),
        "shipping_delay": (
            "Dear {customer_name},\n\n"
            "We sincerely apologize for the delay in delivering your order {order_id}. "
            "This is due to {delay_reason}. Your order is now expected to arrive by {new_date}.\n\n"
            "We understand this is frustrating and appreciate your patience.\n\n"
            "Best regards,\nSupport Team"
        ),
        "product_defect": (
            "Dear {customer_name},\n\n"
            "We are very sorry to hear that you received a defective product. "
            "Please send us photos of the defect and your order number {order_id}, "
            "and we will arrange a replacement or full refund immediately.\n\n"
            "We sincerely apologize for this experience.\n\n"
            "Best regards,\nSupport Team"
        ),
        "account_issue": (
            "Dear {customer_name},\n\n"
            "We have identified the issue with your account. Our technical team "
            "is working to resolve it. In the meantime, you can try resetting your "
            "password using the link below, or contact us directly for manual assistance.\n\n"
            "Best regards,\nSupport Team"
        ),
        "general": (
            "Dear {customer_name},\n\n"
            "Thank you for contacting us. We have received your message and will "
            "get back to you within 24 hours with a complete resolution.\n\n"
            "Best regards,\nSupport Team"
        ),
    }

    def get_template(self, intent: str) -> str:
        return self.TEMPLATES.get(intent, self.TEMPLATES["general"])

    def render(self, template: str, variables: Dict[str, str]) -> str:
        result = template
        for k, v in variables.items():
            result = result.replace(f"{{{k}}}", v)
        return result


class GeminiDraftWriter:
    """Uses Gemini Flash to write contextual, personalized replies."""

    SYSTEM_PROMPT = (
        "You are a professional customer support agent writing email replies. "
        "Be empathetic, concise, and solution-focused. "
        "Do not promise specific outcomes you cannot guarantee. "
        "Keep replies under 150 words. Use formal but friendly tone."
    )

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self._model = None
        if GEMINI_AVAILABLE and api_key:
            try:
                genai.configure(api_key=api_key)
                self._model = genai.GenerativeModel(model)
            except Exception as exc:
                logger.warning("Gemini init failed: %s", exc)

    def draft(self, email: EmailMessage,
               classification: Optional[ClassificationResult]) -> str:
        if self._model is not None:
            intent_str = classification.primary_intent.value if classification else "unknown"
            urgency_str = classification.urgency.value if classification else "medium"
            prompt = (
                f"{self.SYSTEM_PROMPT}\n\n"
                f"Customer email subject: {email.subject}\n"
                f"Customer email body:\n{email.body[:800]}\n\n"
                f"Detected intent: {intent_str}\n"
                f"Urgency level: {urgency_str}\n\n"
                "Write a draft reply email body only (no subject line):"
            )
            try:
                resp = self._model.generate_content(prompt)
                return resp.text.strip()
            except Exception as exc:
                logger.error("Gemini draft error: %s", exc)
        return self._stub_draft(email, classification)

    def _stub_draft(self, email: EmailMessage,
                    classification: Optional[ClassificationResult]) -> str:
        intent = classification.primary_intent.value if classification else "general"
        library = ReplyTemplateLibrary()
        template = library.get_template(intent)
        return library.render(template, {
            "customer_name": "Valued Customer",
            "order_id": "your order",
            "status": "being processed",
            "tracking_url": "[tracking link]",
            "delay_reason": "high demand",
            "new_date": "within 3-5 business days",
        })


class CopilotAuditLog:
    """SQLite-backed audit trail for copilot decisions."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS audit_log (
        entry_id TEXT PRIMARY KEY,
        email_id TEXT,
        action TEXT,
        actor TEXT,
        details TEXT,
        timestamp REAL
    );
    CREATE TABLE IF NOT EXISTS drafts (
        draft_id TEXT PRIMARY KEY,
        email_id TEXT,
        subject TEXT,
        body TEXT,
        intent TEXT,
        urgency TEXT,
        status TEXT,
        generated_at REAL,
        reviewed_by TEXT,
        reviewed_at REAL,
        final_body TEXT
    );
    """

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def save_draft(self, draft: ReplyDraft) -> None:
        intent = draft.classification.primary_intent.value if draft.classification else None
        urgency = draft.classification.urgency.value if draft.classification else None
        self.conn.execute(
            "INSERT OR REPLACE INTO drafts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (draft.draft_id, draft.email_id, draft.subject, draft.body,
             intent, urgency, draft.status.value, draft.generated_at,
             draft.reviewed_by, draft.reviewed_at, draft.final_body),
        )
        self.conn.commit()

    def log_action(self, entry: CopilotAuditEntry) -> None:
        self.conn.execute(
            "INSERT INTO audit_log VALUES (?,?,?,?,?,?)",
            (entry.entry_id, entry.email_id, entry.action, entry.actor,
             json.dumps(entry.details), entry.timestamp),
        )
        self.conn.commit()

    def recent_drafts(self, limit: int = 20) -> List[Dict]:
        import pandas as pd
        return pd.read_sql_query(
            f"SELECT * FROM drafts ORDER BY generated_at DESC LIMIT {limit}",
            self.conn,
        ).to_dict(orient="records")


class SupportCopilot:
    """
    Main copilot: classify -> draft -> human review -> log.
    """

    def __init__(self, api_key: Optional[str] = None, db_path: str = ":memory:"):
        self.classifier = SupportEmailClassifier(api_key=api_key) if CLASSIFIER_AVAILABLE else None
        self.draft_writer = GeminiDraftWriter(api_key=api_key)
        self.audit = CopilotAuditLog(db_path=db_path)
        self._draft_counter = 0

    def process_email(self, email: EmailMessage) -> ReplyDraft:
        classification = None
        if self.classifier:
            classification = self.classifier.classify(email)
            logger.info("Classified %s as %s [urgency=%s]",
                         email.email_id,
                         classification.primary_intent.value,
                         classification.urgency.value)

        draft_body = self.draft_writer.draft(email, classification)
        self._draft_counter += 1
        draft_id = f"draft_{email.email_id}_{self._draft_counter}"

        draft = ReplyDraft(
            draft_id=draft_id,
            email_id=email.email_id,
            subject=f"Re: {email.subject}",
            body=draft_body,
            classification=classification,
        )
        self.audit.save_draft(draft)
        self.audit.log_action(CopilotAuditEntry(
            entry_id=f"ae_{draft_id}",
            email_id=email.email_id,
            action="draft_generated",
            actor="copilot",
            details={
                "intent": classification.primary_intent.value if classification else None,
                "urgency": classification.urgency.value if classification else None,
                "requires_review": classification.requires_human_review if classification else True,
            },
        ))
        return draft

    def approve_draft(self, draft_id: str, reviewer: str,
                       edited_body: Optional[str] = None) -> None:
        self.audit.log_action(CopilotAuditEntry(
            entry_id=f"ae_{draft_id}_approved",
            email_id=draft_id,
            action="draft_approved",
            actor=reviewer,
            details={"edited": edited_body is not None},
        ))

    def reject_draft(self, draft_id: str, reviewer: str, reason: str) -> None:
        self.audit.log_action(CopilotAuditEntry(
            entry_id=f"ae_{draft_id}_rejected",
            email_id=draft_id,
            action="draft_rejected",
            actor=reviewer,
            details={"reason": reason},
        ))

    def is_urgent_alert(self, classification: ClassificationResult) -> bool:
        return (classification.urgency in (UrgencyLevel.CRITICAL, UrgencyLevel.HIGH)
                and classification.primary_intent in
                {SupportIntent.ESCALATION, SupportIntent.COMPLAINT,
                 SupportIntent.REFUND_REQUEST, SupportIntent.PRODUCT_DEFECT})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    copilot = SupportCopilot()

    test_emails = [
        EmailMessage("e001", "Where is my order?",
                      "Order ORD-XY1234 placed a week ago, no tracking info received yet.",
                      "c1@example.com"),
        EmailMessage("e002", "REFUND NEEDED IMMEDIATELY",
                      "Received a completely damaged item. I want my Rs. 12000 back NOW. "
                      "This is absolutely unacceptable. I will escalate to consumer court.",
                      "angry@example.com"),
        EmailMessage("e003", "Great service!",
                      "Just wanted to share how pleased I am with my recent order. "
                      "Delivery was fast and the product is excellent.",
                      "happy@example.com"),
    ]

    print("Support Inbox Copilot Demo\n")
    for email in test_emails:
        draft = copilot.process_email(email)
        cls = draft.classification
        print(f"Email: {email.subject}")
        if cls:
            print(f"  Intent: {cls.primary_intent.value} | Urgency: {cls.urgency.value} "
                  f"| Sentiment: {cls.sentiment}")
            print(f"  Human review required: {cls.requires_human_review}")
            if copilot.is_urgent_alert(cls):
                print("  *** URGENT ALERT: Escalate to senior support immediately ***")
        print(f"  Draft reply ({len(draft.body)} chars):")
        print("  " + draft.body[:200].replace("\n", "\n  "))
        print()

    print("Recent drafts in audit log:")
    for d in copilot.audit.recent_drafts(3):
        print(f"  {d.get('draft_id')} | {d.get('intent')} | {d.get('urgency')} | {d.get('status')}")
