import os
import uuid
import boto3
from typing import List, Dict, Any
#from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import io
import pandas as pd
import json
from logger_utils import get_logger


class RAGIngestor:
    """
    RAG Document Ingestion Service.
    Handles reading documents from S3, chunking, embedding, and storing in Qdrant.
    """

    def __init__(
        self,
        collection_name: str = "knowledge",
        aws_region: str = None,
        chunk_size: int = 1000,
    ):
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.aws_region = aws_region or os.getenv("AWS_REGION", "us-east-1")
        self.logger = get_logger("RAGIngestor")

        # OpenAI embeddings
        self.embed_model = os.getenv("EMBED_MODEL", "amazon.titan-embed-text-v2:0")
        self.vector_size = 1024
        self.bedrock = boto3.client("bedrock-runtime", region_name=self.aws_region)
       #self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # S3 client
        self.s3 = boto3.client("s3", region_name=self.aws_region)

        # Qdrant client
        # In docker-compose, Qdrant is reachable at http://qdrant:6333
        # For local (non-docker) runs, set QDRANT_URL=http://localhost:6333
        # For Qdrant Cloud:
        #   QDRANT_URL=...
        #   QDRANT_API_KEY=...
        self.qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")

        self.qdrant = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        )

    def create_collection(self) -> bool:
        """
        Create a Qdrant collection to store vectors.
        Returns True if created, False if already exists.
        """
        existing_collections = self.qdrant.get_collections().collections
        existing_names = [c.name for c in existing_collections]

        if self.collection_name in existing_names:
            self.logger.info(f"Collection '{self.collection_name}' already exists")
            return False

        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )
        self.logger.info(f"Collection '{self.collection_name}' created (vector size: {self.vector_size})")
        return True

    def chunk_text(self, text: str) -> List[str]:
        """
        Simple chunking: split text into fixed-size chunks.
        Returns list of text chunks.
        """
        self.logger.info(f"chunking text '{text}'")
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

    def embed_text(self, text: str) -> List[float]:
        """
        Convert text to vector embedding using Amazon Bedrock.
        Returns a float embedding vector.
        """
        #self.logger.info(f"embedding text '{text}'")
        response = self.bedrock.invoke_model(
            modelId=self.embed_model,
            body=json.dumps({
                "inputText": text
            }),
            accept="application/json",
            contentType="application/json",
        )

        response_body = json.loads(response["body"].read())
        return response_body["embedding"]

    def upsert_point(self, id: str, vector: List[float], metadata: Dict[str, Any]):
        """
        Insert or update a vector point in Qdrant.
        """
        point = PointStruct(
            id=id,
            vector=vector,
            payload=metadata,
        )

        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

    def ingest_from_s3(self, bucket: str, prefix: str = "") -> Dict[str, Any]:
        """
        Fetch documents from S3 and store them in Qdrant.
        Returns statistics about the ingestion.
        """
        stats = {
            "files_processed": 0,
            "total_chunks": 0,
            "errors": [],
        }
        self.logger.info(f"/ingesting from s3'")

        try:
            objects = self.s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
            #self.logger.info(f"These are objects {objects}")
            for obj in objects:
                key = obj["Key"]

                # Skip directories
                if key.endswith("/"):
                    continue

                try:
                    obj_data= self.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                    if key.endswith(".parquet"):
                        # df = pd.read_parquet(io.BytesIO(obj_data))

                        # body = "\n".join(
                        #     df.apply(
                        #         lambda row: " | ".join(
                        #             f"{col}: {'' if pd.isna(row[col]) else str(row[col])}"
                        #             for col in df.columns
                        #         ),
                        #         axis=1
                        #     ).tolist()
                        # )
                        x= 0
                    else:
                        self.logger.info(f"Decoding object data for {key}")
                        body = obj_data.decode("utf-8")

                    chunks = self.chunk_text(body)

                    for i, chunk in enumerate(chunks):
                        vector = self.embed_text(chunk)
                        self.upsert_point(
                            str(uuid.uuid4()),
                            vector,
                            {
                                "text": chunk,
                                "s3_key": key,
                                "chunk_index": i,
                            },
                        )

                    stats["files_processed"] += 1
                    stats["total_chunks"] += len(chunks)
                    #self.logger.info(f"Indexed: {key} ({len(chunks)} chunks)")
                except Exception as e:
                    error_msg = f"Failed to process {key}: {str(e)}"
                    self.logger.error(error_msg)
                    stats["errors"].append(error_msg)

        except Exception as e:
            error_msg = f"Failed to list S3 objects: {str(e)}"
            self.logger.error(error_msg)
            stats["errors"].append(error_msg)

        return stats