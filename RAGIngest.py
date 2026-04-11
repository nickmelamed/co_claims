import io
import json
import os
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple

import boto3
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from logger_utils import get_logger


class RAGIngestor:
    """
    RAG Document Ingestion Service (enhanced).
    Handles reading documents from S3, row-level chunking, embedding, and storing in Qdrant.
    """

    def __init__(
        self,
        collection_name: str = "knowledge",
        aws_region: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        csv_chunksize: int = 2000,
        upsert_batch_size: int = 100,
        max_embed_workers: int = 8,
        checkpoint_file: str = ".ingest_checkpoint.json",
    ):
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.csv_chunksize = csv_chunksize
        self.upsert_batch_size = upsert_batch_size
        self.max_embed_workers = max_embed_workers
        self.aws_region = aws_region or os.getenv("AWS_REGION", "us-west-2")
        self.logger = get_logger("RAGIngestorNew")

        # Embeddings
        self.embed_model = os.getenv("EMBED_MODEL", "amazon.titan-embed-text-v2:0")
        self.vector_size = 1024
        self.bedrock = boto3.client("bedrock-runtime", region_name=self.aws_region)

        # S3 client
        self.s3 = boto3.client("s3", region_name=self.aws_region)

        # Qdrant client
        self.qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.qdrant = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        )

        # Optional local checkpoint for resumable ingestion
        self.checkpoint_file = checkpoint_file
        self.checkpoints = self._load_checkpoints()

    def _load_checkpoints(self) -> Dict[str, int]:
        if not os.path.exists(self.checkpoint_file):
            return {}
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, dict):
                return {k: int(v) for k, v in data.items()}
            return {}
        except Exception as exc:
            self.logger.warning(f"Could not load checkpoint file: {exc}")
            return {}

    def _save_checkpoints(self) -> None:
        try:
            with open(self.checkpoint_file, "w", encoding="utf-8") as fp:
                json.dump(self.checkpoints, fp)
        except Exception as exc:
            self.logger.warning(f"Could not save checkpoint file: {exc}")

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
        self.logger.info(
            f"Collection '{self.collection_name}' created (vector size: {self.vector_size})"
        )
        return True

    def chunk_text(self, text: str) -> List[str]:
        """
        Sliding-window chunking with overlap.
        """
        text = (text or "").strip()
        if not text:
            return []

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        step = self.chunk_size - self.chunk_overlap
        chunks: List[str] = []
        for i in range(0, len(text), step):
            chunk = text[i : i + self.chunk_size]
            if chunk:
                chunks.append(chunk)
            if i + self.chunk_size >= len(text):
                break
        return chunks

    def embed_text(self, text: str, max_retries: int = 3) -> List[float]:
        """
        Convert text to vector embedding using Amazon Bedrock.
        Includes lightweight retries for transient API failures.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.bedrock.invoke_model(
                    modelId=self.embed_model,
                    body=json.dumps({"inputText": text}),
                    accept="application/json",
                    contentType="application/json",
                )
                response_body = json.loads(response["body"].read())
                return response_body["embedding"]
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    time.sleep(0.5 * attempt)
        raise RuntimeError(f"Embedding failed after {max_retries} attempts: {last_exc}")

    def upsert_point(self, id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """
        Backward-compatible single-point upsert helper.
        """
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=id, vector=vector, payload=metadata)],
        )

    def upsert_points(self, points: List[PointStruct]) -> None:
        """
        Batch upsert helper.
        """
        if not points:
            return
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    @staticmethod
    def _safe_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        return str(value).strip()

    @staticmethod
    def _is_noise_text(text: str) -> bool:
        normalized = " ".join((text or "").strip().lower().split())
        if not normalized:
            return True
        noisy_exact = {
            "stock quotes, and market data",
            "stock quotes and market data",
        }
        return normalized in noisy_exact

    def _row_to_document(
        self,
        row: Dict[str, Any],
        key: str,
        row_index: int,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Build one row-level document text + metadata.
        """
        evidence_text = self._safe_value(row.get("evidence_text"))
        text = self._safe_value(row.get("text"))
        title = self._safe_value(row.get("title"))

        # Prioritize evidence_text; otherwise fallback to text + title.
        doc_text = evidence_text or text
        if not doc_text and title:
            doc_text = title
        if title and title not in doc_text:
            doc_text = f"{title}\n\n{doc_text}".strip()

        if self._is_noise_text(doc_text):
            return None

        payload = {
            "s3_key": key,
            "row_index": row_index,
            "ticker": self._safe_value(row.get("ticker")),
            "company_name": self._safe_value(row.get("company_name")),
            "fact_type": self._safe_value(row.get("fact_type")),
            "direction": self._safe_value(row.get("direction")),
            "filing_date": self._safe_value(row.get("filing_date")),
            "source_url": self._safe_value(row.get("source_url")),
            "news_site": self._safe_value(row.get("news_site")),
            "language": self._safe_value(row.get("language")),
            "sourcecountry": self._safe_value(row.get("sourcecountry")),
            "partition_0": self._safe_value(row.get("partition_0")),
        }
        return doc_text, payload

    def _iter_s3_keys(self, bucket: str, prefix: str = "") -> Iterable[str]:
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("/"):
                    yield key

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        worker_count = max(1, min(self.max_embed_workers, len(texts)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            vectors = list(executor.map(self.embed_text, texts))
        return vectors

    def _flush_pending_points(
        self,
        pending: List[Tuple[str, Dict[str, Any]]],
        stats: Dict[str, Any],
        max_retries: int = 3,
    ) -> None:
        if not pending:
            return

        try:
            texts = [item[0] for item in pending]
            payloads = [item[1] for item in pending]
            vectors = self._embed_batch(texts)
            points = [
                PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)
                for vec, payload in zip(vectors, payloads)
            ]

            last_exc: Optional[Exception] = None
            for attempt in range(1, max_retries + 1):
                try:
                    self.upsert_points(points)
                    stats["total_points"] += len(points)
                    break
                except Exception as exc:
                    last_exc = exc
                    wait = 2 ** attempt
                    self.logger.warning(
                        f"Qdrant upsert attempt {attempt}/{max_retries} failed "
                        f"({len(points)} points): {exc} — retrying in {wait}s"
                    )
                    time.sleep(wait)
            else:
                err = f"Failed embedding/upserting batch of {len(pending)} chunks after {max_retries} retries: {last_exc}"
                self.logger.error(err)
                stats["errors"].append(err)

        except Exception as exc:
            err = f"Failed embedding/upserting batch of {len(pending)} chunks: {exc}"
            self.logger.error(err)
            stats["errors"].append(err)
        finally:
            pending.clear()

    def _ingest_csv_from_s3(self, bucket: str, key: str, stats: Dict[str, Any]) -> None:
        obj = self.s3.get_object(Bucket=bucket, Key=key)
        stream = io.TextIOWrapper(obj["Body"], encoding="utf-8", newline="")
        start_row = int(self.checkpoints.get(key, 0))
        current_row = 0
        pending_points: List[Tuple[str, Dict[str, Any]]] = []

        self.logger.info(f"Streaming CSV '{key}' from row {start_row}")

        try:
            for df_chunk in pd.read_csv(
                stream,
                chunksize=self.csv_chunksize,
                dtype=str,
                on_bad_lines="skip",
                engine="python",
            ):
                df_chunk = df_chunk.fillna("")
                records = df_chunk.to_dict(orient="records")

                for row in records:
                    current_row += 1
                    if current_row <= start_row:
                        continue

                    document = self._row_to_document(row=row, key=key, row_index=current_row)
                    if document is None:
                        stats["skipped_rows"] += 1
                        continue

                    doc_text, base_payload = document
                    chunks = self.chunk_text(doc_text)
                    if not chunks:
                        stats["skipped_rows"] += 1
                        continue

                    stats["rows_processed"] += 1
                    stats["total_chunks"] += len(chunks)
                    for chunk_index, chunk in enumerate(chunks):
                        payload = dict(base_payload)
                        payload["chunk_index"] = chunk_index
                        payload["text"] = chunk
                        pending_points.append((chunk, payload))

                        if len(pending_points) >= self.upsert_batch_size:
                            self._flush_pending_points(pending_points, stats)

                    # Save row-level progress periodically for resumability.
                    if current_row % 1000 == 0:
                        self.checkpoints[key] = current_row
                        self._save_checkpoints()

            self._flush_pending_points(pending_points, stats)
            self.checkpoints[key] = current_row
            self._save_checkpoints()
            self.logger.info(
                f"Finished CSV '{key}': rows={current_row}, points={stats['total_points']}"
            )
        finally:
            stream.close()

    def _ingest_parquet_from_s3(self, bucket: str, key: str, stats: Dict[str, Any]) -> None:
        """
        Parquet fallback path. This still reads into memory once and is kept for compatibility.
        """
        obj_data = self.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        df = pd.read_parquet(io.BytesIO(obj_data)).fillna("")
        pending_points: List[Tuple[str, Dict[str, Any]]] = []

        for idx, row in enumerate(df.to_dict(orient="records"), start=1):
            document = self._row_to_document(row=row, key=key, row_index=idx)
            if document is None:
                stats["skipped_rows"] += 1
                continue

            doc_text, base_payload = document
            chunks = self.chunk_text(doc_text)
            if not chunks:
                stats["skipped_rows"] += 1
                continue

            stats["rows_processed"] += 1
            stats["total_chunks"] += len(chunks)
            for chunk_index, chunk in enumerate(chunks):
                payload = dict(base_payload)
                payload["chunk_index"] = chunk_index
                payload["text"] = chunk
                pending_points.append((chunk, payload))
                if len(pending_points) >= self.upsert_batch_size:
                    self._flush_pending_points(pending_points, stats)

        self._flush_pending_points(pending_points, stats)

    def _ingest_plain_text_from_s3(self, bucket: str, key: str, stats: Dict[str, Any]) -> None:
        """
        Plain text fallback path for non-CSV/non-parquet files.
        """
        text = self.s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        chunks = self.chunk_text(text)
        pending_points: List[Tuple[str, Dict[str, Any]]] = []
        for chunk_index, chunk in enumerate(chunks):
            payload = {"s3_key": key, "chunk_index": chunk_index, "text": chunk}
            pending_points.append((chunk, payload))
        stats["rows_processed"] += 1
        stats["total_chunks"] += len(chunks)
        self._flush_pending_points(pending_points, stats)

    def ingest_from_s3(self, bucket: str, prefix: str = "") -> Dict[str, Any]:
        """
        Fetch documents from S3 and store them in Qdrant.
        Returns statistics about the ingestion.
        """
        stats: Dict[str, Any] = {
            "files_processed": 0,
            "rows_processed": 0,
            "skipped_rows": 0,
            "total_chunks": 0,
            "total_points": 0,
            "errors": [],
        }
        self.logger.info(f"Ingesting from S3 bucket='{bucket}' prefix='{prefix}'")

        try:
            for key in self._iter_s3_keys(bucket, prefix):
                try:
                    key_lower = key.lower()
                    if key_lower.endswith(".csv"):
                        self._ingest_csv_from_s3(bucket, key, stats)
                    elif key_lower.endswith(".parquet"):
                        self._ingest_parquet_from_s3(bucket, key, stats)
                    else:
                        self._ingest_plain_text_from_s3(bucket, key, stats)

                    stats["files_processed"] += 1
                except Exception as exc:
                    error_msg = f"Failed to process {key}: {exc}"
                    self.logger.error(error_msg)
                    stats["errors"].append(error_msg)
        except Exception as exc:
            error_msg = f"Failed to list S3 objects: {exc}"
            self.logger.error(error_msg)
            stats["errors"].append(error_msg)

        return stats
