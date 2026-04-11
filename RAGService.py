import os
import boto3
import asyncio
import traceback
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from RAGIngest import RAGIngestor
from RAGSearch import RAGSearcher
from logger_utils import get_logger

from eval.config import build_pipeline
from eval.judges.client import BedrockClient

from eval.evaluator.deterministic.source_types import classify_source, extract_domain

from datetime import datetime

# Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
LLM_MODEL = os.getenv("LLM_MODEL", "us.amazon.nova-2-lite-v1:0")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")

# Initialize
app = FastAPI(title="RAG AI Search Service", version="1.0.0")
llm = BedrockClient(LLM_MODEL)
logger = get_logger("RAGService")


#Pipeline 

pipeline = build_pipeline()

def parse_timestamp(ts):
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(ts)
    except:
        return None

# Initialize RAG components (using default index name "knowledge")
ingestor = RAGIngestor(aws_region=AWS_REGION)
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


class ChatResponse(BaseModel):
    query: str
    overview: str
    metrics: dict
    credibility: float
    evidence_counts: dict
    sources: list


@app.post("/ingest")
def ingest(request: IngestRequest, authorized: bool = Depends(verify_auth)):
   """Ingest documents from S3."""
   logger.info(f"Ingesting documents from S3: {request.bucket} {request.prefix}")
   try:
       ingestor.create_collection()
       stats = ingestor.ingest_from_s3(request.bucket, request.prefix)
       return {"status": "success", "statistics": stats}
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

            #  try structured domain
            if news_site:
                domain = news_site.lower()

            # fallback to URL parsing
            elif url:
                domain = extract_domain(url)

            # final fallback
            else:
                domain = "unknown"

            # hardcoding checks for our two kinds of data
            if raw_type in ["10-k", "10k", "10-q", "10q"]:
                source_type = "financial_filing"
            elif raw_type in ["news", "news_article"]:
                source_type = "news_article"
            else:
                # fallback to classifier
                source_type = classify_source(url, text)
            

            evidence_list.append({
                "text": text,
                "timestamp": parse_timestamp(m.get("timestamp")),
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
        print("\n=== RAW EVIDENCE DEBUG ===")
        for e in evidence_list:
            print({
                "url": e.get("url"),
                "domain": e.get("domain"),
                "timestamp": e.get("timestamp"),
                "source_type": e.get("source_type")
    })
        
        if any(m is None for m in matches):
            logger.warning(f"Found None in matches: {matches}")
        
        # debugging metadata
        print("MATCH KEYS:", [list(m.keys()) for m in matches])

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

        # sources
        sources = [
            {
                "file": m.get("s3_key"),
                "score": m.get("score"),
                "chunk_index": m.get("chunk_index"),
                "timestamp": m.get("timestamp")
            }
            for m in (matches or [])
            if isinstance(m, dict)
        ]

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
