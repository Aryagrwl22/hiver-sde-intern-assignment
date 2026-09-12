# Brand Candidate Comparison

## Overview
This report evaluates candidate brands for the AI-support-agent pipeline using the `thoughtvector/customer-support-on-twitter` dataset. The evaluation is based on dataset statistics and a qualitative review of a random sample of real conversations to assess suitability for intent classification, historical retrieval, and grounded reply generation.

## Dataset Evidence (from `reports/brand_statistics.csv`)

| Brand | Outbound Tweets | Customer Authors | Conversations | Est. Usable Data |
| :--- | :--- | :--- | :--- | :--- |
| **AmazonHelp** | 169,840 | 155,445 | 82,534 | High (~80k convos) |
| **AppleSupport**| 106,860 | 106,696 | 80,702 | High (~80k convos) |
| **Uber_Support**| 56,270 | 55,283 | 41,923 | Medium (~40k convos) |
| **SpotifyCares**| 43,265 | 41,734 | 28,280 | Medium (~28k convos) |
| **Delta** | 42,253 | 36,215 | 26,166 | Medium (~26k convos) |

*Note: Usable data is an estimate based on total conversations, though some will be filtered for noise/non-English.*

## Qualitative Conversation Quality Observations

Based on a random sample of actual conversations from `data/raw/twcs.csv`:

### 1. AppleSupport
* **Observations**: Interactions are highly technical and understandable (e.g., "battery dropped to 14%", "autocorrect glitch changing 'I'"). Replies often contain actionable workarounds, specific iOS version mentions (e.g., "update 11.1.1 contains a fix"), and support links.
* **Taxonomy Suitability**: Excellent. Clear, repeating issues (battery, updates, glitches, forgotten password).
* **Grounding/RAG Suitability**: **Strongest**. The historical replies contain actual solutions that an LLM can ground its generation on, rather than just asking for a DM.
* **Risks**: Some issues still require DMs for hardware diagnostics.

### 2. AmazonHelp
* **Observations**: Wide variety of issues including missing packages, delivery delays, and Prime queries. Multi-turn conversations are common.
* **Taxonomy Suitability**: Excellent. Intents like `delivery_delay`, `missing_package`, `item_condition` are very clear.
* **Grounding/RAG Suitability**: Fair. While some policy answers exist, the majority of replies are "Please share your details with us here" links.
* **Risks**: High amount of multi-lingual tweets (Spanish, French, etc.) which adds noise. Replies are heavily template-driven without providing the actual resolution publicly.

### 3. SpotifyCares
* **Observations**: Customers report app crashes, missing songs, and account linking issues (e.g., Hulu). 
* **Taxonomy Suitability**: Good. Clear issues (app bug, feature request, account management).
* **Grounding/RAG Suitability**: Fair to Good. They often explain if a song is unavailable, but for app bugs, they usually ask troubleshooting questions ("Can you let us know the device..."). Good for building an escalation flow.
* **Risks**: Lower volume compared to Apple/Amazon.

### 4. Delta
* **Observations**: Flight delays, lost baggage, booking questions.
* **Taxonomy Suitability**: Good.
* **Grounding/RAG Suitability**: Poor. Almost every resolution requires PII ("Please DM your confirmation number"). The AI agent would mostly just learn to ask for a confirmation number.
* **Risks**: Hard to evaluate grounded generation when the golden response is always "DM me your info".

### 5. Uber_Support
* **Observations**: Uber Eats issues, app GPS glitches, lost items.
* **Taxonomy Suitability**: Good.
* **Grounding/RAG Suitability**: Very Poor. Almost every public reply is exclusively a generic "We're here to help! Send us a DM."
* **Risks**: The model will learn to output a generic DM request for every intent, defeating the purpose of advanced RAG.

---

## Ranked Shortlist
1. **AppleSupport** (Best for RAG and actionable solutions)
2. **AmazonHelp** (Best for volume and clear e-commerce taxonomy)
3. **SpotifyCares** (Good balance of technical troubleshooting and volume)
4. **Delta** (Good for classification, but poor for RAG)
5. **Uber_Support** (Too generic)

## Recommended Brand: **AppleSupport** (RECOMMENDATION ONLY)
**Reasons for Recommendation**: 
For this assignment, the pipeline needs to demonstrate *Retrieval-Augmented Generation (RAG)* and *Grounded Reply Generation*. AppleSupport is the only top candidate that consistently provides public, actionable solutions (e.g., workarounds, update version numbers) rather than just redirecting the user to DMs. It also has a massive volume (~80k conversations), ensuring we have enough data to form a strong intent taxonomy.

**Risks/Limitations**:
* Will need to filter out interactions that require hardware diagnostics.
* Some conversations are highly specific to old iOS versions (e.g., iOS 11), meaning the agent will retrieve historically accurate but temporally outdated solutions.

---
**MANUAL DECISION REQUIRED**: Please review the comparison above and confirm your final brand choice so we can proceed with Phase 2 (Taxonomy creation).
