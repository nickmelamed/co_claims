import os
import boto3
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from RAGIngest import RAGIngestor
from RAGSearch import RAGSearcher
from logger_utils import get_logger

# Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
LLM_MODEL = os.getenv("LLM_MODEL", "us.amazon.nova-2-lite-v1:0")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")

# Initialize
app = FastAPI(title="RAG AI Search Service", version="1.0.0")
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
logger = get_logger("RAGService")

# Initialize RAG components (using default index name "knowledge")
ingestor = RAGIngestor(aws_region=AWS_REGION)
searcher = RAGSearcher(collection_name="knowledge", aws_region=AWS_REGION)


def verify_auth(authorization: str = Header(None)):
   """Verify authorization token."""
   if AUTH_TOKEN and authorization != f"Bearer {AUTH_TOKEN}":
       raise HTTPException(status_code=401, detail="Unauthorized")
   return True


class IngestRequest(BaseModel):
   bucket: Optional[str] = "capstone-data"
   prefix: Optional[str] = "rag-files/"


class ChatRequest(BaseModel):
   query: str
   top_k: Optional[int] = 5
   max_tokens: Optional[int] = 500
   temperature: Optional[float] = 0.7


class ChatResponse(BaseModel):
   query: str
   answer: str
   sources: list
   context_used: str


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
   """Answer questions using RAG."""
   try:
       # Search for relevant context
       matches = searcher.search_vectors(request.query, limit=request.top_k)
       context = searcher.format_context(matches)
      
       # Create prompt
       system_prompt = "You are a helpful AI assistant. Answer based on the provided context."
       user_prompt = f"Context:\n{context}\n\nQuestion: {request.query}\n\nAnswer:"
      
       # Call Bedrock
       response = bedrock.converse(
           modelId=LLM_MODEL,
           messages=[{"role": "user", "content": [{"text": user_prompt}]}],
           inferenceConfig={"maxTokens": request.max_tokens, "temperature": request.temperature},
           system=[{"text": system_prompt}]
       )
      
       answer = response['output']['message']['content'][0]['text']
       sources = [{"file": m["s3_key"], "score": m["score"], "chunk_index": m["chunk_index"]} for m in matches]
      
       return ChatResponse(query=request.query, answer=answer, sources=sources, context_used=context)
   except Exception as e:
       logger.error(f"Error answering question: {str(e)}")
       raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
   """Health check."""
   bedrock = boto3.client("bedrock", region_name="us-west-2")
   return {"service": "ok", "region": AWS_REGION, "model": bedrock.list_foundation_models()}


if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)
