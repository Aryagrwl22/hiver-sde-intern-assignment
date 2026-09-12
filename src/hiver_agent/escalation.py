def evaluate_escalation_rules(intent: str, intent_confidence: str, retrieved_evidence: list, customer_text: str) -> tuple[str, str]:
    """
    Evaluates explicit deterministic safety/business rules for obvious escalation cases.
    Returns (decision, reason) or (None, None) if LLM should decide.
    """
    if intent in ["account_and_billing", "hardware_issue"]:
        return "ESCALATE_TO_HUMAN", f"Business Rule: {intent} requires manual verification or physical diagnosis."
        
    if intent_confidence.lower() == "low":
        return "ESCALATE_TO_HUMAN", "Business Rule: Low intent classification confidence."
        
    if intent == "other":
        return "ESCALATE_TO_HUMAN", "Business Rule: Insufficient customer context (intent is 'other')."
        
    if not retrieved_evidence or len(retrieved_evidence) == 0:
        return "ESCALATE_TO_HUMAN", "Business Rule: Insufficient retrieval evidence to generate a grounded reply."
        
    # Simple check for sensitive info keywords
    sensitive_keywords = ["credit card", "ssn", "social security", "password", "bank account", "cvv"]
    lower_text = customer_text.lower()
    if any(keyword in lower_text for keyword in sensitive_keywords):
        return "ESCALATE_TO_HUMAN", "Business Rule: Message contains potentially sensitive/private information."
        
    return None, None
