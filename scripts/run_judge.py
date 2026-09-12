import os
import sys
import pandas as pd
import json
import argparse
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
import numpy as np
import time

sys.path.insert(0, str(Path("c:/Hiver1/src")))
from hiver_agent.retrieval import retrieve
from hiver_agent.judge import evaluate_reply_with_judge

def prepare_samples():
    print("Preparing 25 examples for human judgment...")
    eval_csv = Path("c:/Hiver1/reports/agent_evaluation_official.csv")
    if not eval_csv.exists():
        print(f"Error: {eval_csv} not found.")
        return
        
    df = pd.read_csv(eval_csv)
    
    # Deterministically sample 25
    sample_df = df.sample(n=25, random_state=42).copy()
    
    samples = []
    for _, row in sample_df.iterrows():
        # Retrieve evidence deterministically
        evidence = retrieve(row['customer_text'], top_k=3)
        
        samples.append({
            "tweet_id": row['tweet_id'],
            "customer_text": row['customer_text'],
            "true_intent": row['true_intent'],
            "decision": row['decision'],
            "draft_reply": row['draft_reply'],
            "retrieved_evidence": json.dumps(evidence),
            "human_score": ""
        })
        
    out_df = pd.DataFrame(samples)
    out_dir = Path("c:/Hiver1/data/judge")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "judge_samples.csv"
    
    out_df.to_csv(out_path, index=False)
    print(f"Saved 25 samples to {out_path}.")
    print("ACTION REQUIRED: Please open this file and manually enter a 'human_score' (1-5) for each row.")

def evaluate_judge():
    print("Running LLM judge and calculating agreement...")
    sample_csv = Path("c:/Hiver1/data/judge/judge_samples.csv")
    if not sample_csv.exists():
        print(f"Error: {sample_csv} not found. Run --prepare first.")
        return
        
    df = pd.read_csv(sample_csv)
    
    if 'human_score' not in df.columns or df['human_score'].isna().any():
        print("PENDING: Human scores are missing. Please enter 'human_score' (1-5) for all 25 rows in data/judge/judge_samples.csv")
        return
        
    # Convert to int, dropping any weird spaces
    df['human_score'] = pd.to_numeric(df['human_score'], errors='coerce')
    if df['human_score'].isna().any():
         print("PENDING: Some human scores are invalid. Must be integers 1-5.")
         return
         
    df['human_score'] = df['human_score'].astype(int)
    
    results = []
    for idx, row in df.iterrows():
        print(f"Judging {idx+1}/25 (tweet_id: {row['tweet_id']})...")
        evidence = json.loads(row['retrieved_evidence'])
        
        judge_out = evaluate_reply_with_judge(
            customer_text=row['customer_text'],
            agent_reply=row['draft_reply'],
            retrieved_evidence=evidence,
            agent_decision=row['decision']
        )
        
        results.append({
            "tweet_id": row['tweet_id'],
            "customer_text": row['customer_text'],
            "human_score": row['human_score'],
            "llm_correctness": judge_out.get('correctness_score', 0),
            "llm_groundedness": judge_out.get('groundedness_score', 0),
            "llm_helpfulness": judge_out.get('helpfulness_score', 0),
            "llm_safety": judge_out.get('safety_score', 0),
            "llm_overall_score": judge_out.get('overall_score', 0),
            "llm_reasoning": judge_out.get('reasoning', '')
        })
        time.sleep(2) # Respect Groq rate limits
        
    res_df = pd.DataFrame(results)
    reports_dir = Path("c:/Hiver1/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    res_df.to_csv(reports_dir / "judge_results.csv", index=False)
    
    # Calculate metrics
    human_scores = res_df['human_score'].values
    llm_scores = res_df['llm_overall_score'].values
    
    # Filter out failures (0) if any
    valid_idx = (llm_scores > 0) & (llm_scores <= 5)
    human_valid = human_scores[valid_idx]
    llm_valid = llm_scores[valid_idx]
    
    if len(human_valid) == 0:
        print("Error: LLM returned no valid scores.")
        return
        
    kappa = cohen_kappa_score(human_valid, llm_valid, weights='quadratic')
    exact_agreement = np.mean(human_valid == llm_valid)
    within_one = np.mean(np.abs(human_valid - llm_valid) <= 1)
    
    metrics = {
        "weighted_cohen_kappa": float(kappa),
        "exact_agreement": float(exact_agreement),
        "within_one_agreement": float(within_one),
        "samples_evaluated": int(len(human_valid))
    }
    
    with open(reports_dir / "judge_agreement.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Markdown report
    md = f"""# LLM-as-Judge Evaluation Report

## Agreement Metrics
- **Weighted Cohen's Kappa:** {kappa:.3f}
- **Exact Agreement:** {exact_agreement*100:.1f}%
- **Within-One Agreement:** {within_one*100:.1f}%
- **Samples Evaluated:** {len(human_valid)}/25

## Quality Averages
- **Human Avg Score:** {human_valid.mean():.2f}/5
- **LLM Avg Score:** {llm_valid.mean():.2f}/5
"""
    with open(reports_dir / "judge_evaluation.md", "w") as f:
        f.write(md)
        
    print("\n--- LLM JUDGE EVALUATION COMPLETE ---")
    print(f"Weighted Cohen's Kappa: {kappa:.3f}")
    print(f"Exact Agreement: {exact_agreement*100:.1f}%")
    print(f"Within-One Agreement: {within_one*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Prepare the 25 sample dataset for human rating.")
    parser.add_argument("--evaluate", action="store_true", help="Run the LLM judge and calculate agreement.")
    args = parser.parse_args()
    
    if args.prepare:
        prepare_samples()
    elif args.evaluate:
        evaluate_judge()
    else:
        print("Please specify --prepare or --evaluate")
