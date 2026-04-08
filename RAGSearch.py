import os
import json
import boto3
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from logger_utils import get_logger


class RAGSearcher:
    """
    RAG Search Service.
    Handles embedding queries and searching Qdrant for relevant documents.
    """

    def __init__(
        self,
        collection_name: str = "knowledge",
        aws_region: str = None
    ):
        self.collection_name = collection_name
        self.aws_region = aws_region or os.getenv("AWS_REGION", "us-east-1")
        self.logger = get_logger("RAGSearcher")

        # Bedrock embeddings
        self.embed_model = os.getenv("EMBED_MODEL", "amazon.titan-embed-text-v2:0")
        self.bedrock = boto3.client("bedrock-runtime", region_name=self.aws_region)

        # Qdrant client
        # In docker-compose, Qdrant is reachable at http://qdrant:6333
        # For local (non-docker) runs, set QDRANT_URL=http://localhost:6333
        self.qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")

        self.qdrant = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        )

    def embed_query(self, query: str) -> List[float]:
        """Convert query text to vector embedding using Amazon Bedrock."""
        self.logger.info(f"we make it here to the embed_query function")
        response = self.bedrock.invoke_model(
            modelId=self.embed_model,
            body=json.dumps({
                "inputText": query,
                "dimensions": 1024,
                "normalize": True
            }),
            accept="application/json",
            contentType="application/json",
        )

        response_body = json.loads(response["body"].read())
        return response_body["embedding"]

    def search_vectors(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search Qdrant for similar vectors.
        Returns top matching chunks with their metadata.
        """
        query_vector = self.embed_query(query)

        self.logger.info(f"we make it here to the query_points function")
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        )
        matches = []
        points = results.points
        self.logger.info(f"These are points {points}")
        print(f"These are points {points}")

        for point in points:
            payload = point.payload or {}
            matches.append(
                {
                    "text": payload.get("text", ""),
                    "score": point.score,
                    "s3_key": payload.get("s3_key", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "filing_date": payload.get("filing_date", ""),
                    "fact_type": payload.get("fact_type", ""),
                    "source_url": payload.get("source_url", ""),
                }
            )
        return matches

    def format_context(self, matches: List[Dict[str, Any]]) -> str:
        """Format search results into a context string for LLM."""
        if not matches:
            return "No relevant context found."

        context_parts = []
        for i, match in enumerate(matches, 1):
            context_parts.append(
                f"[Source {i}] (Score: {match['score']:.3f} | File: {match['s3_key']} | "
                f"Type: {match['fact_type']} | Date: {match['filing_date']} | "
                f"URL: {match['source_url']})\n"
                f"{match['text']}\n"
            )

        return "\n".join(context_parts)