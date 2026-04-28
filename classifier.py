"""
Email intent classifier for the support inbox copilot.
Detects intent, urgency, and sentiment from incoming customer emails.
"""
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class SupportIntent(str, Enum):
    ORDER_STATUS = "order_status"
    REFUND_REQUEST = "refund_request"
    SHIPPING_DELAY = "shipping_delay"
    PRODUCT_DEFECT = "product_defect"
    ACCOUNT_ISSUE = "account_issue"
    PAYMENT_ISSUE = "payment_issue"
    GENERAL_INQUIRY = "general_inquiry"
    COMPLAINT = "complaint"
    CANCELLATION = "cancellation"
    FEEDBACK = "feedback"
    ESCALATION = "escalation"
    UNKNOWN = "unknown"


class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EmailMessage:
    email_id: str
    subject: str
    body: str
    sender_email: str
    received_at: float = field(default_factory=time.time)
    thread_id: Optional[str] = None
    customer_id: Optional[str] = None


@dataclass
class ClassificationResult:
    email_id: str
    primary_intent: SupportIntent
    secondary_intent: Optional[SupportIntent]
    urgency: UrgencyLevel
    sentiment: str
    confidence: float
    key_entities: Dict[str, List[str]]
    requires_human_review: bool
    classification_ms: float
    method: str


class RuleBasedClassifier:
    """Fast intent classification using regex patterns."""

    INTENT_PATTERNS: Dict[SupportIntent, List[str]] = {
        SupportIntent.ORDER_STATUS: [
            r"\b(order status|where is my order|track|tracking|shipped)\b",
            r"\b(has my order|order number|when will|delivery date)\b",
        ],
        SupportIntent.REFUND_REQUEST: [
            r"\b(refund|money back|reimburs|return my money)\b",
            r"\b(want a refund|request refund|full refund|partial refund)\b",
        ],
        SupportIntent.SHIPPING_DELAY: [
            r"\b(delayed|late|not arrived|still waiting|overdue)\b",
            r"\b(expected|promised|delivery window|hasn.t arrived)\b",
        ],
        SupportIntent.PRODUCT_DEFECT: [
            r"\b(broken|defective|damaged|not working|malfunction|faulty)\b",
            r"\b(stopped working|dead on arrival|doa|quality issue)\b",
        ],
        SupportIntent.ACCOUNT_ISSUE: [
            r"\b(can.t login|cannot access|locked out|password|account blocked)\b",
            r"\b(reset password|forgot password|account suspended|verify)\b",
        ],
        SupportIntent.PAYMENT_ISSUE: [
            r"\b(charged twice|double charge|payment failed|not charged|billing)\b",
            r"\b(invoice|receipt|payment method|card declined)\b",
        ],
        SupportIntent.CANCELLATION: [
            r"\b(cancel|cancellation|don.t want|no longer need|stop order)\b",
        ],
        SupportIntent.COMPLAINT: [
            r"\b(terrible|worst|awful|unacceptable|disgusting|pathetic)\b",
            r"\b(very disappointed|extremely frustrated|horrible experience)\b",
        ],
        SupportIntent.ESCALATION: [
            r"\b(manager|supervisor|escalate|legal action|consumer court|complaint department)\b",
            r"\b(report|file complaint|going to|media|social media|tweet)\b",
        ],
        SupportIntent.FEEDBACK: [
            r"\b(feedback|suggestion|improve|idea|loved|great experience)\b",
        ],
    }

    URGENCY_PATTERNS: Dict[UrgencyLevel, List[str]] = {
        UrgencyLevel.CRITICAL: [
            r"\b(urgent|emergency|immediately|asap|critical|now|today)\b",
            r"\b(legal action|court|lawsuit|fraud|scam|threatening)\b",
        ],
        UrgencyLevel.HIGH: [
            r"\b(escalate|manager|supervisor|disappointed|unacceptable)\b",
            r"\b(not working|broken|damaged|defect|refund)\b",
        ],
        UrgencyLevel.MEDIUM: [
            r"\b(delayed|late|not arrived|issue|problem|concern)\b",
        ],
    }

    POSITIVE_WORDS = {"great", "love", "excellent", "happy", "satisfied", "wonderful", "perfect"}
    NEGATIVE_WORDS = {"terrible", "awful", "bad", "hate", "angry", "frustrated", "disappointed",
                       "horrible", "worst", "broken", "failed", "useless"}

    def classify_intent(self, text: str) -> Tuple[SupportIntent, float]:
        text_lower = text.lower()
        scores: Dict[SupportIntent, int] = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    scores[intent] = scores.get(intent, 0) + 1
        if not scores:
            return SupportIntent.UNKNOWN, 0.4
        best = max(scores, key=lambda k: scores[k])
        conf = min(0.95, 0.5 + scores[best] * 0.15)
        return best, conf

    def classify_urgency(self, text: str) -> UrgencyLevel:
        text_lower = text.lower()
        for level in [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH, UrgencyLevel.MEDIUM]:
            patterns = self.URGENCY_PATTERNS.get(level, [])
            if any(re.search(p, text_lower) for p in patterns):
                return level
        return UrgencyLevel.LOW

    def detect_sentiment(self, text: str) -> str:
        words = set(text.lower().split())
        pos = len(words & self.POSITIVE_WORDS)
        neg = len(words & self.NEGATIVE_WORDS)
        if neg > pos:
            return "negative"
        if pos > neg:
            return "positive"
        return "neutral"

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        entities: Dict[str, List[str]] = {}
        order_ids = re.findall(r"\b(?:ORD|ORDER|#)[-]?(\w{6,12})\b", text, re.IGNORECASE)
        if order_ids:
            entities["order_ids"] = order_ids
        emails = re.findall(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", text)
        if emails:
            entities["emails"] = emails
        phones = re.findall(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b", text)
        if phones:
            entities["phones"] = phones
        amounts = re.findall(r"(?:Rs\.?|INR|rupees?)\s*[\d,]+(?:\.\d{2})?", text, re.IGNORECASE)
        if amounts:
            entities["amounts"] = amounts
        return entities


class GeminiClassifier:
    """LLM-powered classification for complex or ambiguous emails."""

    CLASSIFICATION_PROMPT = """Analyze this customer support email and return a JSON object with:
- intent: one of {intents}
- urgency: one of low|medium|high|critical
- sentiment: positive|negative|neutral
- confidence: 0.0-1.0

Email subject: {{subject}}
Email body: {{body}}

Return ONLY valid JSON, no explanation."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self._model = None
        if GEMINI_AVAILABLE and api_key:
            try:
                genai.configure(api_key=api_key)
                self._model = genai.GenerativeModel(model)
            except Exception as exc:
                logger.warning("Gemini init failed: %s", exc)

    def classify(self, email: EmailMessage) -> Optional[Dict[str, Any]]:
        if self._model is None:
            return None
        intents = "|".join(i.value for i in SupportIntent)
        prompt = (self.CLASSIFICATION_PROMPT
                  .replace("{intents}", intents)
                  .replace("{{subject}}", email.subject[:200])
                  .replace("{{body}}", email.body[:1000]))
        try:
            resp = self._model.generate_content(prompt)
            text = resp.text.strip()
            text = re.sub(r"```json\s*|\s*```", "", text)
            return json.loads(text)
        except Exception as exc:
            logger.error("Gemini classification error: %s", exc)
            return None


class SupportEmailClassifier:
    """
    Hybrid classifier: fast rule-based first pass, LLM for low-confidence cases.
    """

    LLM_CONFIDENCE_THRESHOLD = 0.6
    HUMAN_REVIEW_INTENTS = {SupportIntent.ESCALATION, SupportIntent.COMPLAINT,
                             SupportIntent.REFUND_REQUEST}
    HUMAN_REVIEW_URGENCIES = {UrgencyLevel.CRITICAL, UrgencyLevel.HIGH}

    def __init__(self, api_key: Optional[str] = None):
        self.rule_classifier = RuleBasedClassifier()
        self.llm_classifier = GeminiClassifier(api_key=api_key)

    def classify(self, email: EmailMessage) -> ClassificationResult:
        t0 = time.perf_counter()
        full_text = f"{email.subject} {email.body}"

        intent, confidence = self.rule_classifier.classify_intent(full_text)
        urgency = self.rule_classifier.classify_urgency(full_text)
        sentiment = self.rule_classifier.detect_sentiment(full_text)
        entities = self.rule_classifier.extract_entities(full_text)
        method = "rule_based"
        secondary = None

        if confidence < self.LLM_CONFIDENCE_THRESHOLD:
            llm_result = self.llm_classifier.classify(email)
            if llm_result:
                try:
                    intent = SupportIntent(llm_result.get("intent", "unknown"))
                    urgency = UrgencyLevel(llm_result.get("urgency", "medium"))
                    sentiment = llm_result.get("sentiment", sentiment)
                    confidence = float(llm_result.get("confidence", 0.75))
                    method = "llm"
                except (ValueError, KeyError):
                    pass

        requires_review = (
            intent in self.HUMAN_REVIEW_INTENTS
            or urgency in self.HUMAN_REVIEW_URGENCIES
            or sentiment == "negative" and urgency != UrgencyLevel.LOW
        )

        ms = (time.perf_counter() - t0) * 1000
        return ClassificationResult(
            email_id=email.email_id,
            primary_intent=intent,
            secondary_intent=secondary,
            urgency=urgency,
            sentiment=sentiment,
            confidence=round(confidence, 4),
            key_entities=entities,
            requires_human_review=requires_review,
            classification_ms=round(ms, 1),
            method=method,
        )

    def classify_batch(self, emails: List[EmailMessage]) -> List[ClassificationResult]:
        return [self.classify(e) for e in emails]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    classifier = SupportEmailClassifier()

    test_emails = [
        EmailMessage("e001", "Where is my order ORD-ABC123?",
                      "I placed an order 5 days ago but haven't received any tracking update. "
                      "My order number is ORD-ABC123. Please help.",
                      "customer1@example.com"),
        EmailMessage("e002", "URGENT - Defective product received",
                      "I received a completely broken phone. This is unacceptable. "
                      "I want a full refund immediately or I will take legal action. "
                      "I paid Rs. 45,000 for this.",
                      "angry.customer@example.com"),
        EmailMessage("e003", "Feedback on recent purchase",
                      "Just wanted to say your customer service was wonderful. "
                      "The delivery was on time and the product quality is excellent. Thank you!",
                      "happy.user@example.com"),
        EmailMessage("e004", "Can't login to my account",
                      "I've been trying to reset my password since yesterday but the email "
                      "link isn't working. My account seems to be locked.",
                      "locked.user@example.com"),
    ]

    print("Support Email Classifier Demo\n")
    for email in test_emails:
        result = classifier.classify(email)
        print(f"Email: {email.subject[:50]}")
        print(f"  Intent: {result.primary_intent.value} ({result.confidence:.0%})")
        print(f"  Urgency: {result.urgency.value} | Sentiment: {result.sentiment}")
        print(f"  Human review: {result.requires_human_review} | Method: {result.method}")
        if result.key_entities:
            print(f"  Entities: {result.key_entities}")
        print(f"  Latency: {result.classification_ms:.1f}ms\n")
