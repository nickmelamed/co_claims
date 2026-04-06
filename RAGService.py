import os
import boto3
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from RAGIngest import RAGIngestor
from RAGSearch import RAGSearcher
from logger_utils import get_logger

from eval.config import build_pipeline
from eval.judges.client import BedrockClient

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
def chat(request: ChatRequest, authorized: bool = Depends(verify_auth)):
    try:
        # retrieval 
        matches = searcher.search_vectors(request.query, limit=request.top_k)
        context = searcher.format_context(matches)

        # evidence list construction 
        evidence_list = []
        for m in matches:
            evidence_list.append({
                "text": m.get("text", ""),  # MUST exist in your vector store
                "timestamp": m.get("timestamp"),
                "source_type": m.get("source_type", "unknown"),
                "score": m.get("score")
            })

        # eval pipeline 
        result = pipeline.run(request.query, evidence_list)

        metrics = result["metrics"]
        credibility = result["credibility"]

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
            for m in matches
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

        overview = llm.chat(
         overview_prompt,
         temperature=0.3,
         max_tokens=150
      )

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
        logger.error(f"Error in /chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
   """Health check."""
   bedrock = boto3.client("bedrock", region_name="us-west-2")
   return {"service": "ok", "region": AWS_REGION, "model": bedrock.list_foundation_models()}


if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)
