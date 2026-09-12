# Hiver SDE Intern Take-Home — Advanced AI Customer Support Agent

This repository implements a complete Generative AI pipeline for customer support ticket classification and auto-resolution, fulfilling the Hiver intern assignment. 

## 1. Problem Framing
The system acts as an autonomous tier-1 support agent on Twitter. It classifies incoming customer tweets into a specific intent, determines whether the issue can be safely auto-handled or must be escalated to a human, and drafts a grounded reply based on historical company knowledge.

**What this system intentionally does NOT attempt to build:**
- We do not attempt to construct a multi-brand mega-classifier. The focus is strictly on one brand to mimic a realistic, specialized enterprise support environment.
- We do not mix in out-of-domain datasets (like Banking77) to artificially inflate training data.
- We do not allow the LLM to invent unsupported troubleshooting policies. Auto-handled troubleshooting must be grounded in the retrieved historical evidence.
- We do not allow the LLM to control high-risk actions. Sensitive intents such as hardware repair and account/billing issues trigger deterministic escalation to a human, preventing automatic resolution for these categories.

## 2. Dataset & Brand Selection
**Dataset:** [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (`twcs.csv`). 

**Brand Selected:** `@AppleSupport`.  
AppleSupport was selected because it provides a large, highly usable set of technical and actionable customer-support conversations, with enough repeated issue patterns to build a robust semantic retrieval index.

## 3. Intent Taxonomy
The current AppleSupport taxonomy was custom-built after analyzing real tweet distributions. It contains 6 intents:
1. **`software_issue`**: iOS bugs, app crashes, abnormal battery drain.
2. **`hardware_issue`**: Physical damage, broken screens, accessory failure.
3. **`account_and_billing`**: Apple ID, iCloud login, Activation Lock, App Store billing.
4. **`how_to_and_inquiry`**: General product inquiries, how to use features, warranty questions.
5. **`complaint_feedback`**: Explicit complaints, negative feedback without troubleshooting requests.
6. **`other`**: Vague messages, gratitude, or unsupported languages.

## 4. Final Architecture
The pipeline implements **Retrieval-Augmented Generation (RAG)** coupled with an LLM router:
1. **Semantic Retrieval:** The incoming customer query is embedded using a local model. A vector search retrieves the top-K most semantically similar historical conversations (evidence).
2. **LLM Intent Classification & Assessment:** An LLM reads the customer query and retrieved evidence, classifies the intent (1 of 6), and outputs a confidence score via structured JSON.
3. **Deterministic Escalation Rules:** Hardcoded safety rules run first. If the intent is `hardware_issue` or `account_and_billing`, or if the LLM confidence is low/evidence is missing, the ticket is forced to `ESCALATE_TO_HUMAN`.
4. **Grounded Reply Generation:** If the ticket is deemed safe to `AUTO_HANDLE`, the LLM drafts a reply that must be strictly grounded in the retrieved historical evidence.

### Retrieval Implementation
We use a completely local embedding model: `sentence-transformers/all-MiniLM-L6-v2`. 
- **Why?** We eliminated the dependency on external embedding APIs (like Gemini) to prevent API rate limit exhaustion and ensure 100% offline, cost-free, and high-speed semantic search.
- The retrieval index is built **exclusively from the `train` split** of the AppleSupport conversations, preventing leakage.

## 5. Evaluation & Official Results
To evaluate the system, we manually constructed a **250-example Golden Evaluation Set** by hand-labeling 250 customer queries from the `test` and `val` splits. 

To strictly prevent data leakage:
- The 250 evaluation rows were excluded from any baseline training data.
- The TF-IDF baseline was trained using a proxy weak-supervision method on the remaining `train` split.

All three systems below were evaluated strictly against the exact same **250 human-labelled golden examples**.

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| **Majority Class Baseline** | 10.4% | 0.031 |
| **TF-IDF + Logistic Regression** | 56.8% | 0.472 |
| **LLM Agent (`qwen3.8-27b`)** | **75.0%** | **0.640** |

*(Note: These are the official, human-verified results. Do not confuse them with stale development metrics).*

## 6. LLM-As-Judge Evaluation
To evaluate the quality of the generated replies, we implemented an LLM-as-Judge using a 1-5 scalar rubric measuring Correctness, Grounding, Helpfulness, and Safety. We took a sample of 25 agent replies and compared human grades to the LLM judge's grades.

**Agreement Results:**
- **Weighted Cohen's Kappa:** 0.079
- **Exact Agreement:** 16%
- **Within-One Agreement:** 76%

**Conclusion:** The agreement is extremely weak (Kappa near zero). While the judge's score usually lands within 1 point of the human, it cannot reliably differentiate between a "good" reply and a "great" reply. Therefore, the judge should **not** be treated as a standalone reliable metric for this project.

## 7. Failure Analysis
Out of 250 examples, the LLM Agent misclassified 63 intents. The top 5 confusion patterns were:

1. **`software_issue` -> `hardware_issue` (14 errors)**
   - *Example:* "Battery life realy short down 50% at least per day, have shut off many features. Still poor battery life @applesupport 11.0.3 battery"
   - *Hypothesis:* The customer mentioned an iOS update causing the issue, making it a software bug. The LLM zeroed in on the word "battery" and falsely assumed physical hardware failure.
2. **`software_issue` -> `complaint_feedback` (10 errors)**
   - *Example:* "nah mate, phones going proper slow and it’s pissing me right off. @115858 get ur shit together"
   - *Hypothesis:* The angry tone overwhelms the LLM. Humans correctly isolated "phones going proper slow" as a software issue, but the LLM classified it purely on sentiment.
3. **`other` -> `complaint_feedback` (5 errors)**
   - *Example:* "Dear @115858, can you please not mess up my phone ? I’m a broke college student & cant afford a new one ."
   - *Hypothesis:* A vague plea labeled as `other` by a human because it lacks context. The LLM categorizes it as a general complaint.
4. **`other` -> `software_issue` (4 errors)**
   - *Example:* "What the HEY! .@115858 .@AppleSupport went for my morning walk and can’t listen to my DLed podcasts and music w/out an internet connection? 👎"
   - *Hypothesis:* Vague phrasing labeled `other` by the human, but the LLM overconfidently assumes it's an iOS Music app bug.
5. **`how_to_and_inquiry` -> `software_issue` (4 errors)**
   - *Example:* A user asking how to downgrade from iOS 11 back to iOS 10.
   - *Hypothesis:* The user is asking a "how to" question, but because it involves an OS update, the LLM mistakes it for reporting a software bug.

## 8. What is misleading about my headline number?
While 75.0% accuracy / 0.64 Macro F1 sounds solid, it has major limitations:
1. **Tiny Sample Size:** It is evaluated on only 250 examples, which is insufficient to prove statistical reliability at scale.
2. **Single Brand:** The system was only tested on `@AppleSupport`. It may fail completely on an airline or banking dataset.
3. **Subjective Taxonomy:** The 6 intents were defined specifically for this project. In reality, enterprise support requires hundreds of granular intents.
4. **Ignoring Reply Quality:** The 75% accuracy metric ONLY scores intent classification. Reply quality was only judged on 25 examples, and as shown above, human-judge agreement was extremely weak.

## 9. Next Week / Future Improvements
- **Few-Shot Prompting:** The LLM intent prompt is currently zero-shot. Adding representative failure examples as few-shot demonstrations could reduce the hardware-vs-software and tone-related classification errors.
- **Separate Sentiment Extraction:** Run a separate sentiment classification pass before intent classification to prevent angry tones from hijacking the intent router.
- **Better Judge Rubric:** The LLM-as-Judge needs a vastly simplified binary rubric (Pass/Fail) rather than a 1-5 scale to improve human agreement.

## 10. Decision Log
Here are 12 non-obvious decisions made during this project:
1. **Hardcoded AppleSupport:** Picked because it has a large, highly usable set of conversations with technical and actionable support issues, making it well-suited for retrieval-based support.
2. **Dropped Banking77:** Blending distinct domains compromises the realism of brand-specific customer support.
3. **Switched to Groq (`qwen/qwen3.8-27b`):** Gemini free-tier suffered from frequent 503 errors during bulk evaluations.
4. **Structured JSON (Pydantic):** Used Groq's OpenAI-compatible endpoint to guarantee perfectly parsable intent classifications.
5. **Manual Retry-Backoff:** Implemented explicit handling of `RateLimitError` to respect API headers and prevent crashed evaluations.
6. **Local Embeddings:** Migrated from remote Gemini embeddings to local `sentence-transformers/all-MiniLM-L6-v2` to eliminate API failures and dimension mismatches entirely.
7. **Strict Conversation-Level Splits:** Split data by conversation ID rather than tweet ID to prevent information leakage (e.g., a brand reply in train hinting at a customer intent in test).
8. **Proxy Baseline Training:** Trained the TF-IDF baseline entirely via keyword heuristics (`hint_intent`) on the train split to guarantee zero leakage from the human golden set.
9. **Deterministic Escalation Rules:** Hardcoded `hardware_issue` and `account_and_billing` to bypass LLM auto-replies, ensuring safety compliance without trusting the LLM.
10. **Evaluation Checkpointing:** Added CSV checkpointing to the evaluation loop so API interruptions wouldn't lose already-processed rows.
11. **Deterministic Golden Sampling:** Sampled test examples deterministically (`Seed=42`) to ensure reproducible baseline comparisons.
12. **Unified 1-5 Judge Metric:** Simplified the LLM-as-judge prompt to output a 1-5 scale across Correctness, Grounding, Helpfulness, and Safety to easily calculate Cohen's Kappa against human scores.

## 11. Reproducibility & Output Files

**Output Paths:**
- Golden Labels: `data/golden/golden_AppleSupport.csv`
- Official Agent Eval: `reports/agent_evaluation_official.csv`
- Official Agent Metrics: `reports/agent_metrics_official.json`
- TF-IDF Metrics: `reports/tfidf_logreg_golden_metrics.json`
- Majority Metrics: `reports/majority_golden_metrics.json`
- Judge Results: `reports/judge_results.csv` and `reports/judge_agreement.json`

### Fast Verification (Read existing artifacts)
You do not need to re-run the 250 LLM calls to verify the results. You can view the exact outputs in the `reports/` directory.

### Full API Evaluation (Live Run)
*Warning: Running the full live evaluation requires a Groq API Key and takes several minutes due to rate limit backoffs.*

1. **Setup Environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```
2. **Add API Key:** Create a `.env` file in the root directory and add `GROQ_API_KEY=your_key`.
3. **Run Official Evaluation:**
   ```powershell
   python scripts/run_agent.py --evaluate
   ```
4. **Run Judge Evaluation:**
   ```powershell
   python scripts/run_judge.py --evaluate
   ```
