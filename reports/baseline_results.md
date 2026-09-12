# Baseline Evaluation Results

## Trivial Baseline Methodology
The trivial baseline predicts the most frequent intent class from the training set (the "Majority Class") for every single prediction. It does not look at the input text at all. This establishes the absolute minimum performance floor. For this dataset, the majority class is `software_issue`.

## TF-IDF + Logistic Regression Methodology
This is a standard traditional Machine Learning baseline. 
1. **TF-IDF (Term Frequency-Inverse Document Frequency)**: Converts the customer's text into a numerical vector based on word counts, penalizing common words and rewarding rare but informative words.
2. **Logistic Regression**: A linear model that learns weights for these word vectors to predict the probability of each intent class.

## Data Split & Leakage
The evaluation uses a **conversation-level split**. Tweets from the same thread are strictly kept together in either train, validation, or test. This prevents data leakage where the model memorizes specific customer usernames, timestamps, or unique phrases from a conversation in the training set and artificially inflates performance on the evaluation set.

## Evaluation Methodology
* **Metrics**: Accuracy, Macro F1, Per-class Precision, Recall, and F1.
* **Golden Set**: PENDING HUMAN LABELING. The official metrics cannot be claimed until a human annotates the 250 evaluation rows. 

## Development Proxy Results (WARNING: PROXY LABELS)
To verify the pipeline works, we ran the baselines on the validation set using our `keyword_hint` as proxy labels. **These are NOT ground-truth human labels and cannot be used as the final headline result.**

**Majority Baseline (Proxy)**
* Accuracy: 46.0%
* Macro F1: ~0.10

**TF-IDF + LogReg (Proxy)**
* Accuracy: 54.0%
* Macro F1: 0.26

## Limitations & Expected Role
These baselines struggle with the nuance, varied vocabulary, and contextual ambiguity of customer support tweets. They serve to prove that a simple bag-of-words model is insufficient for high-quality support routing. We expect the advanced LLM classifier in Phase 4 to significantly outperform these baselines, particularly in the Macro F1 score on minority classes like `hardware_issue` and `how_to_and_inquiry`.
