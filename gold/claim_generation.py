import json
from typing import List, Dict
import cohere
import pickle
import random 
import numpy as np
import os 
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from dotenv import load_dotenv
load_dotenv()

# config 
MODES = ["well_supported", "contradictory", "unsupported", "ambiguous"]

CLAIM_PROMPT = """
You are generating evaluation claims for a financial reasoning system.

Company:
{company}

Evidence:
{evidence}

Task:
Generate ONE claim that is {mode} relative to the evidence.

Modes:
- well_supported: clearly supported by most evidence
- contradictory: directly contradicted by the evidence
- unsupported: plausible but NOT mentioned in the evidence
- ambiguous: partially supported, mixed or unclear

Rules:
- The claim must be realistic and finance-related
- Use specific metrics if possible (revenue, earnings, margins, etc.)
- Do NOT copy text directly from evidence
- Keep it concise (1 sentence)
- Do NOT hallucinate facts outside the evidence except for "unsupported"

Output:
<claim>
Return ONLY the claim as plain text. No explanations.
"""


# LLM call 
def call_llm(prompt: str, co_client) -> str:
    response = co_client.chat(
        message=prompt,
        model="command-a-03-2025", 
        temperature=0.4,
        max_tokens=100
    )
    
    return response.text.strip()


# evidence preparation 
def format_evidence(evidence: List[Dict], max_chars=500) -> str:
    texts = []
    for e in evidence:
        text = e.get("text", "")[:max_chars]
        texts.append(f"- {text}")
    
    return "\n".join(texts)


# validation script 
def is_valid_claim(claim: str) -> bool:
    if not claim:
        return False
    
    claim = claim.strip()
    
    return (
        len(claim) > 20 and
        len(claim) < 300 and
        not claim.lower().startswith("the evidence") and
        "not mentioned" not in claim.lower()
    )


# generate claim for a single mode 
def generate_claim(example: Dict, mode: str, llm) -> str:
    evidence_text = format_evidence(example["evidence"])
    
    prompt = CLAIM_PROMPT.format(
        company=example["company_name"],
        evidence=evidence_text,
        mode=mode
    )
    
    claim = call_llm(prompt, llm)
    
    if is_valid_claim(claim):
        return claim
    
    return None


# generate the full claim set 
def generate_gold_dataset(examples: List[Dict], llm) -> List[Dict]:
    gold_dataset = []
    
    for ex in examples:
        for mode in MODES:
            claim = generate_claim(ex, mode, llm)
            
            if not claim:
                continue
            
            gold_dataset.append({
                "ticker": ex["ticker"],
                "company_name": ex["company_name"],
                "claim": claim,
                "label": mode,

                # original evidence (LLM-friendly)
                "evidence": ex["evidence"],

                # structured view (metric-friendly)
                "evidence_structured": {
                    "texts": [e["text"] for e in ex["evidence"]],
                    "dates": [e["date"] for e in ex["evidence"]],
                    "source_types": [e["source_type"] for e in ex["evidence"]],
                    "urls": [e["url"] for e in ex["evidence"]],
                },

                "metadata": ex.get("metadata", {})
            })
    
    return gold_dataset


# save
def save_gold_dataset(gold_dataset: List[Dict], path="./gold/gold_dataset.json"):

    def clean_types(obj):
        if isinstance(obj, dict):
            return {k: clean_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_types(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        else:
            return obj
        
    cleaned = clean_types(gold_dataset)

    with open(path, "w") as f:
        json.dump(cleaned, f, indent=2)


#run everything in conjunction 
def main():
    with open('./gold/data/examples.pkl', 'rb') as file:
        examples = pickle.load(file)

    # load only a subset of examples for smaller call 
    example_subset = random.sample(examples, 10)

    llm = cohere.Client(os.getenv("CO_API_KEY"))

    gold = generate_gold_dataset(example_subset, llm)
    save_gold_dataset(gold)

if __name__ == "__main__":
    main()
