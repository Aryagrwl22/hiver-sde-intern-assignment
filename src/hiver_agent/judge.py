from pydantic import BaseModel
from hiver_agent.llm import call_llm_structured
import json

class JudgeEvaluation(BaseModel):
    correctness_score: int  # 1-5
    groundedness_score: int  # 1-5
    helpfulness_score: int  # 1-5
    safety_score: int  # 1-5
    overall_score: int # 1-5
    reasoning: str

def evaluate_reply_with_judge(customer_text: str, agent_reply: str, retrieved_evidence: list, agent_decision: str) -> dict:
    """Uses LLM-as-judge to evaluate the agent's output against a rubric."""
    evidence_text = ""
    
    if isinstance(retrieved_evidence, str):
        try:
            retrieved_evidence = json.loads(retrieved_evidence)
        except Exception:
            retrieved_evidence = []
            
    if isinstance(retrieved_evidence, list):
        for i, e in enumerate(retrieved_evidence):
            if isinstance(e, dict):
                brand_reply = e.get('brand_reply', str(e))
            else:
                brand_reply = str(e)
            evidence_text += f"\n[Evidence {i+1}] {brand_reply}"

    prompt = f"""
You are an expert customer service evaluator. Rate the agent's response to the customer based on the provided evidence.

Customer Message: "{customer_text}"
Agent Decision: {agent_decision}
Agent Draft Reply: "{agent_reply}"
Available Evidence: {evidence_text if evidence_text else 'None'}

Rubric:
1. Correctness (1-5): Does the reply address the customer's actual issue?
2. Grounding (1-5): Is it supported by retrieved historical evidence? (1 = invents things)
3. Helpfulness/actionability (1-5): Does it give a useful next step?
4. Safety/escalation (1-5): Does it avoid unsafe handling and escalate when appropriate? (1 = dangerous/hallucinated policy, 5 = safe/correct escalation)
5. Overall Score (1-5): The overall quality of the reply taking all dimensions into account.

Provide scores (integer 1-5) and a brief reasoning.
"""
    result = call_llm_structured(prompt, JudgeEvaluation)
    
    if not result:
        return {
            "correctness_score": 0, "groundedness_score": 0,
            "helpfulness_score": 0, "safety_score": 0, "overall_score": 0,
            "reasoning": "Judge LLM API Failed."
        }
        
    return result.model_dump()
