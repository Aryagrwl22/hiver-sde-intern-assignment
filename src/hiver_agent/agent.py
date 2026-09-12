from typing import Literal, Optional
from pydantic import BaseModel
from hiver_agent.retrieval import retrieve
from hiver_agent.escalation import evaluate_escalation_rules
from hiver_agent.llm import call_llm_structured

class AgentResponse(BaseModel):
    intent: Literal["software_issue", "hardware_issue", "account_and_billing", "how_to_and_inquiry", "complaint_feedback", "other"]
    intent_confidence: Literal["high", "medium", "low"]
    decision: Literal["AUTO_HANDLE", "ESCALATE_TO_HUMAN"]
    escalation_reason: Optional[str]
    draft_reply: str

def run_agent(customer_text: str) -> dict:
    """
    Conceptual Flow:
    1. Semantic retrieval of evidence.
    2. Intent classification & evidence assessment via LLM.
    3. Deterministic Escalation checks based on intent and rules.
    4. Grounded reply generation via LLM.
    
    For efficiency, we can do 2, 3, and 4 in a single LLM call if we inject the deterministic rules,
    but the prompt requires us to explicitly enforce deterministic safety rules in python.
    So we'll use the LLM to classify and draft, but override the decision locally if rules match.
    """
    
    # 1. Semantic Retrieval
    evidence = retrieve(customer_text, top_k=3)
    
    # Format evidence for prompt
    evidence_text = ""
    for i, e in enumerate(evidence):
        evidence_text += f"\n--- Historical Example {i+1} ---\nCustomer: {e['customer_text']}\nBrand Reply: {e['brand_reply']}\n"
    
    if not evidence_text:
        evidence_text = "No historical evidence found."

    prompt = f"""
You are an advanced AppleSupport customer support agent.
Analyze the following customer message and determine its intent based on our taxonomy.
Then, using the provided historical evidence, draft a grounded reply.
Do NOT hallucinate troubleshooting steps. Rely strictly on the historical evidence.

Taxonomy:
- software_issue: bugs, app crashes, freezing, ios updates
- hardware_issue: physical device damage, broken buttons, repair requests
- account_and_billing: Apple ID, Activation Lock, billing, Apple Pay
- how_to_and_inquiry: how to use features, warranty info
- complaint_feedback: rants, negative feedback
- other: vague, gratitude, non-english

Customer Message:
"{customer_text}"

Historical Evidence:
{evidence_text}

Provide:
1. The intent.
2. Confidence level (high, medium, low).
3. A draft reply grounded ONLY in the historical evidence. If evidence is insufficient to confidently answer, draft a generic polite message.
4. Your recommended decision (AUTO_HANDLE or ESCALATE_TO_HUMAN).
5. If escalating, a brief reason.
"""

    llm_result = call_llm_structured(prompt, AgentResponse)
    
    # 10. ERROR HANDLING (API Failure Fallback)
    if not llm_result:
        return {
            "intent": "other",
            "intent_confidence": "low",
            "retrieved_evidence": evidence,
            "decision": "ESCALATE_TO_HUMAN",
            "escalation_reason": "API Failure or Invalid LLM Output.",
            "draft_reply": "Apologies, we are experiencing technical difficulties. Connecting you to a human agent."
        }
        
    # Extract
    intent = llm_result.intent
    confidence = llm_result.intent_confidence
    decision = llm_result.decision
    reason = llm_result.escalation_reason
    reply = llm_result.draft_reply
    
    # 4. DETERMINISTIC ESCALATION OVERRIDE
    override_decision, override_reason = evaluate_escalation_rules(intent, confidence, evidence, customer_text)
    
    if override_decision:
        decision = override_decision
        reason = override_reason
        
    return {
        "intent": intent,
        "intent_confidence": confidence,
        "retrieved_evidence": evidence,
        "decision": decision,
        "escalation_reason": reason,
        "draft_reply": reply
    }
