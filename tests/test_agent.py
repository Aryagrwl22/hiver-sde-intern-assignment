import pytest
from unittest.mock import patch
from hiver_agent.escalation import evaluate_escalation_rules
from hiver_agent.agent import run_agent

def test_escalation_hardware_billing():
    dec, rsn = evaluate_escalation_rules("hardware_issue", "high", [{"msg": "x"}], "my screen broke")
    assert dec == "ESCALATE_TO_HUMAN"
    
    dec, rsn = evaluate_escalation_rules("account_and_billing", "high", [{"msg": "x"}], "payment failed")
    assert dec == "ESCALATE_TO_HUMAN"

def test_escalation_low_confidence():
    dec, rsn = evaluate_escalation_rules("software_issue", "low", [{"msg": "x"}], "it broke")
    assert dec == "ESCALATE_TO_HUMAN"

def test_escalation_insufficient_evidence():
    dec, rsn = evaluate_escalation_rules("software_issue", "high", [], "it broke")
    assert dec == "ESCALATE_TO_HUMAN"

def test_escalation_sensitive_info():
    dec, rsn = evaluate_escalation_rules("software_issue", "high", [{"msg": "x"}], "my password is 123")
    assert dec == "ESCALATE_TO_HUMAN"

@patch('hiver_agent.agent.call_llm_structured')
@patch('hiver_agent.agent.retrieve')
def test_agent_fallback_api_failure(mock_retrieve, mock_call):
    # Simulate API failure returning None
    mock_retrieve.return_value = [{"customer_text": "hello", "brand_reply": "hi"}]
    mock_call.return_value = None
    
    res = run_agent("I need help")
    assert res['decision'] == "ESCALATE_TO_HUMAN"
    assert res['intent'] == "other"
    assert "technical difficulties" in res['draft_reply'].lower()

@patch('hiver_agent.agent.call_llm_structured')
@patch('hiver_agent.agent.retrieve')
def test_agent_override(mock_retrieve, mock_call):
    # Simulate LLM deciding to AUTO_HANDLE but rules override it
    from hiver_agent.agent import AgentResponse
    mock_retrieve.return_value = [{"customer_text": "hi", "brand_reply": "hello"}]
    
    mock_call.return_value = AgentResponse(
        intent="hardware_issue",
        intent_confidence="high",
        decision="AUTO_HANDLE",
        escalation_reason="",
        draft_reply="Try restarting it."
    )
    
    res = run_agent("my screen is cracked")
    assert res['intent'] == "hardware_issue"
    # Overriden by rules:
    assert res['decision'] == "ESCALATE_TO_HUMAN"
