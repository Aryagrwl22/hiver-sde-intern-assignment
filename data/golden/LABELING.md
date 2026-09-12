# AppleSupport Golden Set Labeling Guide

## Introduction
Thank you for helping label the golden evaluation set! Your manual annotations will serve as the ground truth for evaluating our AI customer support routing models. 

**IMPORTANT**: 
* **Do NOT** copy the `keyword_hint` directly. Read the `customer_text` and make your own judgment. The hints are just automated guesses.
* **Do NOT** use external context or guess what the customer means based on what Apple replied. The intent must be derivable from the customer's text alone.

## File Format
Please open `golden_AppleSupport.csv` in Excel or your preferred spreadsheet editor.
Read the `customer_text` column and type the exact intent name into the empty `intent` column.

---

## Valid Intents

You MUST choose exactly one of these six intents for every row.

### 1. `software_issue`
* **What it means**: The customer is experiencing a software bug, an app crash, a device freezing, or abnormal battery drain, specifically often related to updating iOS or macOS.
* **When to choose it**: When the device is physically fine but the software is acting broken or erratic.
* **When NOT to choose it**: When they are just asking *how* to use a feature that is working as intended (use `how_to_and_inquiry`), or if the device is physically broken (use `hardware_issue`).
* **Real Examples**:
  * *"are you guys gonna fix the I glitch?"*
  * *"all this issues started since I update iOs to the 11 version"*

### 2. `hardware_issue`
* **What it means**: The customer's physical device is damaged or a hardware component (like a button, screen, or speaker) is failing. Includes repair requests.
* **When to choose it**: When the root cause is physical hardware failure.
* **When NOT to choose it**: If a feature is broken due to a software update (use `software_issue`).
* **Real Examples**:
  * *"my the volume buttons on my Apple earphones arent working"*
  * *"my screen is cracked, how much is repair?"*

### 3. `account_and_billing`
* **What it means**: Intentionally broad category for sensitive/account-specific issues: Apple ID, iCloud login, Activation Lock, forgotten passwords, App Store billing, and Apple Pay.
* **When to choose it**: Anytime the user mentions payment, login problems, or locked accounts.
* **When NOT to choose it**: If they are asking a generic non-account policy question (use `how_to_and_inquiry`).
* **Real Examples**:
  * *"This activation lock randomly popped up... I forgot my password."*
  * *"debit card was added smoothly to Apple Pay on my watch but not my phone"*

### 4. `how_to_and_inquiry`
* **What it means**: The customer is asking for information, how to use a feature, checking warranty status, or asking about storage space management. No active bug or break is mentioned.
* **When to choose it**: When the customer wants to know how to do something that is functioning correctly, or wants general product info.
* **When NOT to choose it**: If they ask "how to fix" a bug/glitch (use `software_issue` or `hardware_issue`).
* **Real Examples**:
  * *"can I downgrade my phone OS from iOS 11.0 to iOS 10?"*
  * *"do all ur products have a warranty?"*

### 5. `complaint_feedback`
* **What it means**: The customer is solely complaining, venting, or expressing frustration without asking for a specific troubleshooting solution.
* **When to choose it**: When the tweet is purely a rant or negative feedback.
* **When NOT to choose it**: If they complain but *also* specify what is broken and ask for help (use the specific issue category instead).
* **Real Examples**:
  * *"YOU F***ING FIX IT YOU BROKE IT DAMMIT"*
  * *"Apple's new update is the worst thing ever. I hate it."*

### 6. `other`
* **What it means**: Fallback category for messages that cannot be confidently assigned to the defined support intents.
* **When to choose it**: Vague statements ("Help me Apple"), gratitude ("Thanks!"), non-English tweets, or completely unrelated comments.
* **When NOT to choose it**: If there is enough context to guess one of the top 5 intents.
* **Real Examples**:
  * *"Okay thank you...."*
  * *"I finally got a reply to my DM. Thanks!"*
  * *votre mise jour 11.0.3 elle rend fou* (Non-English)

---

## Rules for Ambiguous Messages

* **`software_issue` vs `how_to_and_inquiry`**: If the customer says "How do I fix my phone freezing?" it is a `software_issue` because a bug is occurring. If they say "How do I turn on do not disturb?", it is `how_to_and_inquiry` because the feature isn't broken.
* **`hardware_issue` vs `software_issue`**: If the phone "won't turn on" and the context is immediately after an update, lean towards `software_issue`. If it was dropped or mentions a specific physical component like a button, use `hardware_issue`. When in doubt, pick the most severe root cause.
* **`account_and_billing` vs `how_to_and_inquiry`**: Questions about *how* to use Apple Pay go to `how_to_and_inquiry`. If their specific card is being declined or they are locked out, use `account_and_billing`.
* **`complaint_feedback` vs `other`**: Pure frustration goes to `complaint_feedback`. Short, vague pleas like "Please DM me" or "Help" go to `other`.
* **When to use `other`**: Always use `other` if the message lacks enough context to be actionable, or if it's a follow-up "thank you" tweet in a thread. Do not guess what they need if they just say "It's not working."
