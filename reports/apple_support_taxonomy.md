# AppleSupport Intent Taxonomy

## Methodology
The taxonomy was developed by sampling and analyzing a random set of 200 inbound customer tweets directed at `AppleSupport`. By manually reviewing the customer problems and requests, distinct recurring patterns were identified. The goal was to create a small, defensible set of intents that are mutually exclusive, capture the bulk of the support volume, and are actionable for an AI agent.

## Discovered Intents & Explanations

1. **`software_issue`**
   * **Description**: Software bugs, iOS/macOS update issues, freezing, app crashes, abnormal battery drain.
   * **Why it exists**: The vast majority of AppleSupport tweets relate to software updates (e.g., iOS 11 bugs, the famous 'I' typing glitch, fast battery drain after an update). Grouping all software-related bugs here provides a massive chunk of actionable data for RAG.
   * **Examples**: 
     - "are you guys gonna fix the I glitch?"
     - "all this issues started since I update iOs to the 11 version"

2. **`hardware_issue`**
   * **Description**: Physical device damage, broken buttons, accessory hardware failure, hardware repair requests.
   * **Why it exists**: Hardware issues require distinct handling from software issues. They almost always result in a recommendation to visit an Apple Store or mail in the device.
   * **Examples**:
     - "my the volume buttons on my Apple earphones arent working"

3. **`account_and_billing`**
   * **Description**: Intentionally broad category for sensitive/account-specific issues sharing handling characteristics: Apple ID, iCloud login, Activation Lock, forgotten passwords, App Store billing, and Apple Pay.
   * **Why it exists**: Account security and payments are highly sensitive and heavily regulated. These require PII and secure verification, often mandating human escalation.
   * **Examples**:
     - "This activation lock randomly popped up... I forgot my password."
     - "debit card was added smoothly to Apple Pay on my watch but not my phone"

4. **`how_to_and_inquiry`**
   * **Description**: Asking how to use a feature, queries about warranty, storage space management, or general product inquiries.
   * **Why it exists**: Informational questions that aren't bugs or breaks. These are perfect candidates for automated LLM replies citing Apple documentation.
   * **Examples**:
     - "can I downgrade my phone OS from iOS 11.0 to iOS 10?"
     - "do all ur products have a warranty?"

5. **`complaint_feedback`**
   * **Description**: Explicit complaints, rants, or negative feedback where no clear troubleshooting request is made.
   * **Why it exists**: Separates unactionable rants from actual support requests so the agent doesn't hallucinate troubleshooting steps.
   * **Examples**:
     - "YOU F***ING FIX IT YOU BROKE IT DAMMIT"

6. **`other`**
   * **Description**: Fallback category for messages that cannot be confidently assigned to the defined support intents, including vague/insufficient-context messages, gratitude, and unsupported/non-target-language messages.
   * **Why it exists**: A catch-all for noise, non-English queries, or simple "thank you" follow-ups.

## Merged and Excluded Intents
* **Excluded `refund_cancellation`**: While common for retail, app refunds for Apple are usually handled via a specific portal, and the volume on Twitter is heavily overshadowed by technical issues. These are folded into `account_and_billing`.
* **Merged `ios_update` and `app_crash`**: Merged into `software_issue` because the troubleshooting steps are often identical (force restart, update software, reinstall app).

## Data Leakage & Evaluation Split Strategy
* The dataset was split at the **conversation level** (using `split.py`).
* A tweet's `conversation_id` determines its split. This guarantees that tweets from the same thread do not appear in both the training set (used for RAG/baselines) and the evaluation set, preventing data leakage.
* The golden set is sampled proportionally from train, val, and test splits while maintaining conversation-level isolation.

## Status
* **MANUAL STEP REQUIRED**: The golden evaluation set (`data/golden/golden_candidates_AppleSupport.csv`) must be hand-labeled by humans before running final evaluation metrics. We do not automatically label these using keywords.
