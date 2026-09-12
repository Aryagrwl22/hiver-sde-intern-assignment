import sys
import pandas as pd
from pathlib import Path

def validate():
    golden_path = Path("c:/Hiver1/data/golden/golden_AppleSupport.csv")
    if not golden_path.exists():
        print("ERROR: Golden set file not found.")
        sys.exit(1)
        
    df = pd.read_csv(golden_path)
    
    # Check columns
    required_cols = ["split", "conversation_id", "tweet_id", "customer_text", "keyword_hint", "intent", "labeler"]
    for col in required_cols:
        if col not in df.columns:
            print(f"ERROR: Missing required column '{col}'")
            sys.exit(1)
            
    # Check 150-250 examples
    if not (150 <= len(df) <= 250):
        print(f"ERROR: Golden set must have 150-250 examples, but has {len(df)}")
        sys.exit(1)
        
    # Check duplicates
    if df["tweet_id"].duplicated().any():
        print("ERROR: Duplicate tweet_ids found.")
        sys.exit(1)
        
    # Check missing labels
    missing = df["intent"].isna().sum()
    if missing > 0:
        print(f"ERROR: Found {missing} rows with missing 'intent'.")
        sys.exit(1)
        
    # Check valid intents
    valid_intents = {"software_issue", "hardware_issue", "account_and_billing", "how_to_and_inquiry", "complaint_feedback", "other"}
    invalid = df[~df["intent"].isin(valid_intents)]
    if not invalid.empty:
        print(f"ERROR: Found invalid intents: {invalid['intent'].unique()}")
        sys.exit(1)
        
    print("SUCCESS: Golden set is valid!")
    sys.exit(0)

if __name__ == "__main__":
    validate()
