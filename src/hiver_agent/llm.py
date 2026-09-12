import os
import time
import json
from typing import Any, Optional
from pydantic import BaseModel
from hiver_agent.settings import GROQ_API_KEY, LLM_MODEL, EMBEDDING_MODEL
from openai import OpenAI
import openai
from sentence_transformers import SentenceTransformer

# Setup local embedding model (loaded once globally)
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
except Exception as e:
    print(f"Error loading sentence-transformers model: {e}")
    embedding_model = None

# Setup OpenAI client pointing to Groq
try:
    groq_client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        max_retries=0 # We will handle retries manually to parse retry-after correctly
    )
except Exception:
    groq_client = None

def get_embedding(text: str) -> list[float]:
    """Generates an embedding for the given text using a local sentence-transformer. Returns a list of floats."""
    if not embedding_model:
        return [0.0] * 384 # Fallback to MiniLM dimension if missing
        
    try:
        # encode returns a numpy array, convert to list
        return embedding_model.encode(text).tolist()
    except Exception as e:
        print(f"Warning: Local embedding failed: {e}")
        return [0.0] * 384

def call_llm_structured(prompt: str, response_format: type[BaseModel]) -> Optional[BaseModel]:
    """Calls the LLM and returns the parsed Pydantic object, or None if it fails. Includes retry logic."""
    if not groq_client or not GROQ_API_KEY or GROQ_API_KEY == "mock_key_if_missing":
        return None
        
    max_retries = 5
    for attempt in range(max_retries):
        try:
            try:
                response = groq_client.beta.chat.completions.parse(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "You must respond with valid JSON matching the exact requested schema."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=response_format,
                    temperature=0.0,
                )
                return response.choices[0].message.parsed
            except Exception as parse_error:
                # If parse fails (or isn't supported), fallback to JSON object mode
                if isinstance(parse_error, openai.RateLimitError):
                    raise parse_error # Let outer loop catch rate limits
                    
                response = groq_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "You must respond with valid JSON matching the requested fields."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                content = response.choices[0].message.content
                if content:
                    return response_format.model_validate_json(content)
                else:
                    raise Exception("Empty content returned from fallback JSON mode")
                    
        except openai.RateLimitError as e:
            if attempt < max_retries - 1:
                # Try to extract retry-after from headers, default to 15s if missing
                retry_after = e.response.headers.get('retry-after')
                if retry_after and retry_after.isdigit():
                    wait_time = float(retry_after) + 1.0
                elif retry_after:
                    # Groq sometimes returns strings like '15s'
                    import re
                    match = re.search(r'([\d.]+)s', retry_after)
                    wait_time = float(match.group(1)) + 1.0 if match else 15.0
                else:
                    # Look in the error message for "Please try again in X"
                    import re
                    msg_match = re.search(r'try again in ([\d.]+)s', str(e))
                    wait_time = float(msg_match.group(1)) + 1.0 if msg_match else 15.0
                    
                print(f"Rate limit hit! Waiting {wait_time:.1f}s before retry (Attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"Warning: LLM API failed after {max_retries} attempts: {e}")
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print(f"Warning: LLM API failed after {max_retries} attempts: {e}")
                return None
