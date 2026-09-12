import pandas as pd
from pathlib import Path
import json
import time
import os
from hiver_agent.agent import run_agent
from hiver_foundation.evaluate import compute_metrics, save_eval_artifacts

def evaluate_agent_pipeline():
    print("Running advanced agent evaluation on the golden set...")
    golden_path = Path("c:/Hiver1/data/golden/golden_AppleSupport.csv")
    if not golden_path.exists():
        print("Golden candidates file not found.")
        return
        
    df = pd.read_csv(golden_path)
    
    has_human_labels = df['intent'].notna().sum() == len(df)
    eval_name_suffix = "official" if has_human_labels else "PROXY"
    print(f"Human labels status: {'Complete' if has_human_labels else 'PENDING'}. Using {eval_name_suffix} evaluation mode.")
    
    if not has_human_labels:
        df = df[df['split'] == 'test'].head(10).copy()
        df['eval_label'] = df['keyword_hint']
    else:
        df['eval_label'] = df['intent']
        
    out_path = Path(f"c:/Hiver1/reports/agent_evaluation_{eval_name_suffix}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    # Checkpoint loading
    if out_path.exists():
        try:
            checkpoint_df = pd.read_csv(out_path)
            results = checkpoint_df.to_dict('records')
            print(f"Loaded {len(results)} completed rows from checkpoint.")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
    
    completed_ids = {str(r['tweet_id']) for r in results}
    
    y_true = []
    y_pred = []
    
    total = len(df)
    count = 0
    
    for _, row in df.iterrows():
        count += 1
        tweet_id = str(row['tweet_id'])
        customer_text = row['customer_text']
        true_label = row['eval_label']
        
        y_true.append(true_label)
        
        if tweet_id in completed_ids:
            # Reconstruct y_pred from checkpoint
            existing_row = next((r for r in results if str(r['tweet_id']) == tweet_id), None)
            if existing_row:
                y_pred.append(existing_row['pred_intent'])
            continue
            
        print(f"Evaluating {count}/{total}: {tweet_id}...")
        
        agent_out = run_agent(customer_text)
        
        # Check if we hit an unhandled API error that returned ESCALATE_TO_HUMAN due to a crash
        # Actually run_agent returns ESCALATE_TO_HUMAN on API failure. 
        # But we added 429 logic in llm.py that will pause and retry. If it completely fails, we save the fallback.
        
        pred_label = agent_out['intent']
        y_pred.append(pred_label)
        
        results.append({
            "tweet_id": row['tweet_id'],
            "customer_text": customer_text,
            "true_intent": true_label,
            "pred_intent": pred_label,
            "intent_confidence": agent_out['intent_confidence'],
            "decision": agent_out['decision'],
            "escalation_reason": agent_out['escalation_reason'],
            "draft_reply": agent_out['draft_reply'],
            "judge_correctness": 0,
            "judge_groundedness": 0,
            "judge_hallucinated": False,
            "judge_reasoning": "SKIPPED to avoid API rate limits"
        })
        
        # Save checkpoint after every row
        pd.DataFrame(results).to_csv(out_path, index=False)
        
        # Throttle to respect Groq limits
        time.sleep(2) 
        
    metrics = compute_metrics(y_true, y_pred)
    Path(f"c:/Hiver1/reports/agent_metrics_{eval_name_suffix}.json").write_text(json.dumps(metrics, indent=2))
    
    print("\n--- AGENT EVALUATION RESULTS ---")
    if not has_human_labels:
        print("WARNING: Intent accuracy relies on PROXY keyword hints. Do NOT report as ground truth.")
    print(f"Intent Accuracy ({eval_name_suffix}): {metrics['accuracy']:.2f}")
    print(f"Intent Macro F1 ({eval_name_suffix}): {metrics['macro_f1']:.2f}")
    print("Human-vs-LLM Judge Agreement: PENDING (Requires manual human rating of generated replies)")
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    evaluate_agent_pipeline()
