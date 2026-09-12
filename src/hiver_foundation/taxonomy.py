"""Brand-specific intent taxonomy *framework*.

These labels are a starting point for HAND labeling. They are not gold labels
and are not used as fake evaluation results.
"""

from __future__ import annotations

import json
from pathlib import Path

from hiver_foundation.settings import TAXONOMY_DIR

# Shared, beginner-friendly intents. A brand file can override descriptions
# and keyword *hints* (hints help humans label; they are not predictions).
DEFAULT_INTENTS = [
    {
        "intent": "account_access",
        "description": "Login, password, locked account, two-factor, profile.",
        "keyword_hints": ["login", "password", "account", "locked", "sign in", "username"],
    },
    {
        "intent": "billing_payment",
        "description": "Charges, invoices, payment method, failed payment.",
        "keyword_hints": ["bill", "charged", "payment", "card", "invoice", "refund fee"],
    },
    {
        "intent": "refund_cancellation",
        "description": "Refunds, returns, cancelling a service or order.",
        "keyword_hints": ["refund", "cancel", "return", "money back"],
    },
    {
        "intent": "order_shipping",
        "description": "Where is my order, delayed delivery, tracking, missing package.",
        "keyword_hints": ["order", "shipping", "delivery", "tracking", "package", "late"],
    },
    {
        "intent": "technical_issue",
        "description": "App/site/product not working, bugs, outages, errors.",
        "keyword_hints": ["not working", "error", "bug", "crash", "down", "issue", "broken"],
    },
    {
        "intent": "product_info",
        "description": "How-to questions, features, coverage, policy questions.",
        "keyword_hints": ["how do i", "does it", "policy", "feature", "hours"],
    },
    {
        "intent": "complaint_feedback",
        "description": "Complaints, poor service, compliments that are not a request.",
        "keyword_hints": ["worst", "terrible", "disappointed", "rude", "never again"],
    },
    {
        "intent": "escalation_human",
        "description": "Customer explicitly asks for a human / supervisor / DM with PII.",
        "keyword_hints": ["speak to", "manager", "human", "dm me", "phone"],
    },
    {
        "intent": "other",
        "description": "Does not fit the intents above, or too little text to tell.",
        "keyword_hints": [],
    },
]

BRAND_NOTES = {
    "AppleSupport": "Expect iOS, iCloud, device hardware, App Store billing.",
    "AmazonHelp": "Expect orders, packages, Prime, returns.",
    "SpotifyCares": "Expect playback, playlists, Premium billing.",
    "Uber_Support": "Expect trips, driver issues, receipts, lost items.",
    "Delta": "Expect flights, delays, baggage, miles.",
    "Tesco": "Expect groceries, deliveries, Clubcard, store issues.",
}


def taxonomy_for_brand(brand: str) -> dict:
    if brand == "AppleSupport":
        return {
          "brand": "AppleSupport",
          "notes": "Custom taxonomy for AppleSupport focused on their specific support domain.",
          "intents": [
            {
              "intent": "software_issue",
              "description": "Software bugs, iOS/macOS update issues, freezing, app crashes, abnormal battery drain.",
              "keyword_hints": ["update", "glitch", "crash", "freeze", "bug", "ios", "high sierra", "battery", "type", "slow"]
            },
            {
              "intent": "hardware_issue",
              "description": "Physical device damage, broken buttons, accessory hardware failure, hardware repair requests.",
              "keyword_hints": ["button", "screen", "broken", "repair", "earphones", "hardware", "turn on"]
            },
            {
              "intent": "account_and_billing",
              "description": "Intentionally broad category for sensitive/account-specific issues sharing handling characteristics: Apple ID, iCloud login, Activation Lock, forgotten passwords, App Store billing, and Apple Pay.",
              "keyword_hints": ["password", "apple id", "icloud", "activation lock", "charge", "billing", "apple pay", "login"]
            },
            {
              "intent": "how_to_and_inquiry",
              "description": "Asking how to use a feature, queries about warranty, storage space management, or general product inquiries.",
              "keyword_hints": ["how to", "how do i", "storage", "warranty", "downgrade", "delete"]
            },
            {
              "intent": "complaint_feedback",
              "description": "Explicit complaints, rants, or negative feedback where no clear troubleshooting request is made.",
              "keyword_hints": ["fix it", "terrible", "worst", "annoying", "sucks", "wtf"]
            },
            {
              "intent": "other",
              "description": "Fallback category for messages that cannot be confidently assigned to the defined support intents, including vague/insufficient-context messages, gratitude, and unsupported/non-target-language messages.",
              "keyword_hints": ["thanks", "thank you", "dm", "ok"]
            }
          ],
          "labeling_rule": "Read the customer tweet (and prior tweets in the thread if needed). Pick exactly one intent. If two apply, pick the customer's main request. Use other when unsure. Do not guess from the brand reply alone."
        }

    return {
        "brand": brand,
        "notes": BRAND_NOTES.get(
            brand,
            "Review a sample of this brand's customer tweets and drop/rename intents that never appear.",
        ),
        "intents": DEFAULT_INTENTS,
        "labeling_rule": (
            "Read the customer tweet (and prior tweets in the thread if needed). "
            "Pick exactly one intent. If two apply, pick the customer's main request. "
            "Use other when unsure. Do not guess from the brand reply alone."
        ),
    }



def save_taxonomy(brand: str, path: Path | None = None) -> Path:
    TAXONOMY_DIR.mkdir(parents=True, exist_ok=True)
    dest = path or (TAXONOMY_DIR / f"taxonomy_{_safe_name(brand)}.json")
    payload = taxonomy_for_brand(brand)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


def load_taxonomy(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hint_intent(text: str, taxonomy: dict) -> str:
    """Keyword hint for the labeler. NOT a gold label. Returns 'other' if nothing matches."""
    lowered = (text or "").lower()
    best = "other"
    best_hits = 0
    for item in taxonomy.get("intents", []):
        intent = item["intent"]
        if intent == "other":
            continue
        hits = sum(1 for kw in item.get("keyword_hints", []) if kw.lower() in lowered)
        if hits > best_hits:
            best_hits = hits
            best = intent
    return best if best_hits > 0 else "other"


def _safe_name(brand: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in brand)
