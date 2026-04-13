import os
import boto3
import asyncio
import traceback
import json
import time
from typing import Optional
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from RAG.RAGIngest import RAGIngestor
from RAG.RAGSearch import RAGSearcher
from logger_utils import get_logger

from eval.config import build_pipeline
from eval.judges.client import BedrockClient

from eval.evaluator.deterministic.source_types import classify_source, extract_domain
from RAG.retriever import retrieve_fn, parse_timestamp

from datetime import datetime

# Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
LLM_MODEL = os.getenv("LLM_MODEL", "us.amazon.nova-2-lite-v1:0")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
INGEST_TIMING_FILE = os.getenv("INGEST_TIMING_FILE", "logs/ingest_timing.log")

# Initialize
app = FastAPI(title="RAG AI Search Service", version="1.0.0")
llm = BedrockClient(LLM_MODEL)
logger = get_logger("RAGService")


#Pipeline 

pipeline = build_pipeline(retrieve_fn=retrieve_fn)

# Initialize RAG components (using default index name "knowledge")
ingestor = RAGIngestor(aws_region=AWS_REGION, max_embed_workers=24)
searcher = RAGSearcher(collection_name="knowledge", aws_region=AWS_REGION)


def verify_auth(authorization: str = Header(None)):
   """Verify authorization token."""
   if AUTH_TOKEN and authorization != f"Bearer {AUTH_TOKEN}":
       raise HTTPException(status_code=401, detail="Unauthorized")
   return True


class IngestRequest(BaseModel):
   bucket: Optional[str] = "co-claims-scraped-data"
   prefix: Optional[str] = "mdna_facts_v2_first100.csv"


class ChatRequest(BaseModel):
   query: str
   top_k: Optional[int] = 5
   max_tokens: Optional[int] = 500
   temperature: Optional[float] = 0.7

# Update to make follow-up convo work (start)
class ChatResponse(BaseModel):
    query: str
    overview: str
    metrics: dict
    credibility: float
    evidence_counts: dict
    sources: list

class FollowupRequest(BaseModel):
    original_claim: str
    overview: str
    metrics: dict
    credibility: float
    followup_question: str
    variances: Optional[dict] = {}
    sources: Optional[list] = []

def classify_question(q: str):
    q = q.lower()
    if "why" in q:
        return "why"
    elif "improve" in q or "better" in q:
        return "improve"
    elif "evidence" in q or "source" in q:
        return "evidence"
    elif "score" in q or "metric" in q:
        return "metrics"
    return "general"

def build_evidence_context(sources, max_items=5, max_chars=300):
    formatted = []
    for i, s in enumerate(sources[:max_items]):
        text = (s.get("text") or "")[:max_chars]
        formatted.append(
            f"[{i+1}] {text}\nURL: {s.get('url')}\nScore: {s.get('score')}"
        )
    return "\n\n".join(formatted)

@app.post("/followup")
async def followup(request: FollowupRequest, authorized: bool = Depends(verify_auth)):
    try:
        question_type = classify_question(request.followup_question)

        evidence_context = build_evidence_context(request.sources)

        followup_prompt = f"""
You are an expert assistant explaining the results of a claim evaluation system.

Your job is to answer follow-up questions about:
- the credibility score
- the evaluation metrics
- the retrieved evidence
- the reasoning behind the analysis

Original Claim
{request.original_claim}

Summary of Analysis
{request.overview}

Credibility Score
{request.credibility}

Metrics (0-1 scale)
{request.metrics}

Metric Variances (uncertainty)
{request.variances}

Metric definitions:
- ESS: Evidence Support Score
- ECS: Evidence Contradiction Score
- CMS: Claim Measurability
- LCS: Logical Consistency
- HLS: Hedging Level
- EAS: Evidence Availability
- ERS: Evidence Recency
- ESTS: Evidence Strength
- EAGS: Evidence Agreement
- SRS: Source Diversity
- EVS: External Verifiability
- CScope: Claim Scope

Evidence (Top Retrieved)
{evidence_context}

Question Tyep
{question_type}

Follow-up Question
{request.followup_question}

Instructions

1. ONLY use the provided metrics, summary, and evidence.
2. Do NOT invent facts or evidence.
3. Be concise but insightful.

4. Behavior by question type:
   - WHY:
     Explain the reasoning using both metrics and evidence.
   - METRICS:
     Interpret scores clearly (what is high/low and why).
   - EVIDENCE:
     Reference specific evidence snippets (e.g., [1], [2]).
   - IMPROVE:
     Suggest concrete ways to improve credibility (better evidence, clearer claim, etc.).
   - GENERAL:
     Provide a helpful explanation grounded in the analysis.

5. If uncertainty is high (variance is large), mention it.

6. If information is missing, say so explicitly.

Respond conversationally.
"""

        reply = await asyncio.to_thread(llm.chat, followup_prompt, 0.5, 400)

        return {
            "reply": reply,
            "question_type": question_type
        }

    except Exception as e:
        logger.error("FULL TRACEBACK:\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def parse_evidence_text(raw_text: str):
    if not raw_text:
        return {"clean_text": None, "section": None, "topic": None, "sentiment": None}

    parts = raw_text.split(",")

    # Extract metadata safely
    section = parts[-4] if len(parts) >= 4 else None
    topic = parts[-3] if len(parts) >= 3 else None
    sentiment = parts[-2] if len(parts) >= 2 else None

    # only extract quoted text if it exists
    if '"' in raw_text:
        clean_text = raw_text.split('"')[-2].strip()
    else:
        clean_text = None  

    return {
        "section": section,
        "topic": topic,
        "sentiment": sentiment,
        "clean_text": clean_text
    }

def _append_ingest_timing(entry: dict) -> None:
   """Append one ingest timing entry as JSONL."""
   try:
       timing_path = INGEST_TIMING_FILE
       timing_dir = os.path.dirname(timing_path)
       if timing_dir:
           os.makedirs(timing_dir, exist_ok=True)
       with open(timing_path, "a", encoding="utf-8") as fp:
           fp.write(json.dumps(entry) + "\n")
   except Exception as exc:
       logger.warning(f"Failed to write ingest timing entry: {exc}")


@app.post("/ingest")
def ingest(request: IngestRequest, authorized: bool = Depends(verify_auth)):
   """Ingest documents from S3."""
   logger.info(f"Ingesting documents from S3: {request.bucket} {request.prefix}")
   try:
       started_at = datetime.now(timezone.utc)
       start_monotonic = time.monotonic()
       ingestor.create_collection()
       stats = ingestor.ingest_from_s3(request.bucket, request.prefix)
       elapsed_seconds = round(time.monotonic() - start_monotonic, 3)
       finished_at = datetime.now(timezone.utc)

       timing_entry = {
           "started_at_utc": started_at.isoformat(),
           "finished_at_utc": finished_at.isoformat(),
           "duration_seconds": elapsed_seconds,
           "bucket": request.bucket,
           "prefix": request.prefix,
           "statistics": stats,
       }
       _append_ingest_timing(timing_entry)

       return {
           "status": "success",
           "statistics": stats,
           "ingest_duration_seconds": elapsed_seconds,
           "ingest_timing_file": INGEST_TIMING_FILE,
       }
   except Exception as e:
       logger.error(f"Error ingesting documents: {str(e)}")
       raise HTTPException(status_code=500, detail=str(e))
   


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, authorized: bool = Depends(verify_auth)):
    try:
        # retrieval 
        matches = searcher.search_vectors(request.query, limit=request.top_k)

        if not matches:
            matches = []

        matches = [
            m for m in (matches or [])
            if isinstance(m, dict)
        ]

        logger.info(f"MATCHES RAW: {matches}")

        context = searcher.format_context(matches)

        # evidence list construction 
        evidence_list = []
        for m in matches or []:
            if not isinstance(m, dict):
                continue

            url = m.get("source_url", "")
            text = m.get("text", "")
            raw_type = m.get("fact_type", "").lower()
            news_site = m.get("news_site", "")

            timestamp = m.get("timestamp") or m.get("filing_date")


            # hardcoding checks for our two kinds of data
            if raw_type in ["10-k", "10k", "10-q", "10q"]:
                source_type = "financial_filing"
            elif raw_type in ["news", "news_article"]:
                source_type = "news_article"
            else:
                # fallback to classifier
                source_type = classify_source(url, text)

            news_site = m.get("news_site", "")

            if news_site:
                domain = news_site.lower()

            elif source_type == "financial_filing":
                domain = "sec.gov"

            elif url:
                domain = extract_domain(url)

            else:
                domain = "unknown"

            evidence_list.append({
                "text": text,
                "timestamp": parse_timestamp(timestamp),
                "source_type": source_type,
                "score": m.get("score", 0.0),
                "url": url,
                "domain": domain
            })

        if not evidence_list:
            return {
                "metrics": {},
                "variances": {},
                "credibility": 0.0,
                "decision": {"decision": "no_evidence"},
                "structured": {},
                "entities": []
            }
        
        # debugging domain/dates
    #     print("\n=== RAW EVIDENCE DEBUG ===")
    #     for e in evidence_list:
    #         print({
    #             "url": e.get("url"),
    #             "domain": e.get("domain"),
    #             "timestamp": e.get("timestamp"),
    #             "source_type": e.get("source_type")
    # })
        
        if any(m is None for m in matches):
            logger.warning(f"Found None in matches: {matches}")
        
        # debugging metadata
        #print("MATCH KEYS:", [list(m.keys()) for m in matches])

        # eval pipeline 
        result = await pipeline.run(request.query, evidence_list)

        logger.info(f"PIPELINE OUTPUT TYPE: {type(result)}")
        logger.info(f"PIPELINE OUTPUT: {result}")

        # result guardrails 
        if result is None:
            raise ValueError("Pipeline returned None")
        
        if not isinstance(result, dict):
            logger.error(f"BAD RESULT TYPE: {type(result)} | VALUE: {result}")
            raise ValueError("Pipeline returned invalid result")

        if result.get("metrics") is None:
            logger.error(f"RESULT MISSING METRICS: {result}")
            result["metrics"] = {}

        if result.get("credibility") is None:
            logger.error(f"RESULT MISSING CREDIBILITY: {result}")
            result["credibility"] = 0.0

        metrics = result.get("metrics", {}) or {}
        credibility = result.get("credibility", {}) or {} 

        # evidence counts
        n = len(evidence_list)
        support = int(metrics.get("ESS", 0) * n)
        contradict = int(metrics.get("ECS", 0) * n)

        sources = []
        for m in (matches or []):
            if not isinstance(m, dict):
                continue

            parsed = parse_evidence_text(m.get("text"))

        sources = [
            {
                "file": m.get("s3_key"),
                "url": m.get("source_url"),
                "score": m.get("score"),
                "text": parsed['clean_text'], 
                "chunk_index": m.get("chunk_index"),
                "timestamp": m.get("timestamp")
            }
            for m in (matches or [])
            if isinstance(m, dict)
        ]

        # last check to make sure source has evidence text

        sources = [s for s in sources if s.get("text")]

        # LLM overview 
        overview_prompt = f"""
You are summarizing a claim evaluation.

Claim:
{request.query}

Metrics:
{metrics}

Credibility Score:
{credibility}

Write a concise 2-3 sentence summary explaining:
- Whether the claim is credible
- Whether evidence supports or contradicts it
- Any uncertainty
"""

        overview = await asyncio.to_thread(
        llm.chat,
        overview_prompt,
        0.3,
        150
      )
        
        def clean_metrics(m):
            return {k: float(v) if hasattr(v, "item") else v for k, v in m.items()}
        
        metrics = clean_metrics(metrics) # cleaning outputs for ChatResponse
        credibility = float(credibility)

        # structured output 
        return ChatResponse(
            query=request.query,
            overview=overview,
            metrics=metrics,
            credibility=credibility,
            evidence_counts={
                "supporting": support,
                "contradicting": contradict
            },
            sources=sources
        )

    except Exception as e:
        logger.error("FULL TRACEBACK:\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
   """Health check."""
   bedrock = boto3.client("bedrock", region_name="us-west-2")
   return {"service": "ok", "region": AWS_REGION, "model": bedrock.list_foundation_models()}


if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)
