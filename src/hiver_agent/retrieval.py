import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from hiver_agent.llm import get_embedding
from hiver_agent.settings import RETRIEVAL_INDEX_SIZE
from hiver_foundation.load import load_tweets
from hiver_foundation.conversations import reconstruct_conversations

INDEX_FILE = Path("c:/Hiver1/data/processed/retrieval_index.json")

def build_retrieval_index(csv_path: str = "c:/Hiver1/data/raw/twcs.csv", max_size: int = RETRIEVAL_INDEX_SIZE):
    if INDEX_FILE.exists():
        print("Retrieval index already exists. Skipping build.")
        return
        
    print("Building retrieval index...")
    conv_path = Path("c:/Hiver1/data/processed/conversations.csv")
    
    if not conv_path.exists():
        print("Processed data not found. Please run foundation pipeline first.")
        return
        
    conv = pd.read_csv(conv_path)
    
    # Filter to only train split, and has brand text
    train_conv = conv[(conv['split'] == 'train') & 
                      (conv['first_brand_text'].notna())].copy()
                      
    index_data = []
    sampled_conv = train_conv.head(max_size)
    
    count = 0
    for _, row in sampled_conv.iterrows():
        conv_id = row['conversation_id']
        cust_text = str(row['first_customer_text'])
        brand_reply = str(row['first_brand_text'])
        
        emb = get_embedding(cust_text)
        
        index_data.append({
            "conversation_id": int(conv_id),
            "customer_text": cust_text,
            "brand_reply": brand_reply,
            "embedding": emb
        })
        
        count += 1
        
    # Save to disk
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps(index_data), encoding="utf-8")
    print(f"Saved {len(index_data)} indexed conversations.")

def retrieve(query_text: str, top_k: int = 3) -> list[dict]:
    if not INDEX_FILE.exists():
        print("Warning: Retrieval index not found. Returning empty evidence.")
        return []
        
    index_data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    if not index_data:
        return []
        
    query_emb = get_embedding(query_text)
    
    # Compute cosine similarity
    embs = np.array([item['embedding'] for item in index_data])
    q_emb = np.array([query_emb])
    
    sims = cosine_similarity(q_emb, embs)[0]
    
    # Get top k indices
    top_indices = sims.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        item = index_data[idx]
        results.append({
            "customer_text": item["customer_text"],
            "brand_reply": item["brand_reply"],
            "similarity": float(sims[idx])
        })
        
    return results
