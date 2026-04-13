import os
from datetime import datetime, timezone
from RAG.RAGSearch import RAGSearcher
from eval.evaluator.deterministic.source_types import classify_source, extract_domain

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
searcher = RAGSearcher(collection_name="knowledge", aws_region=AWS_REGION)


def parse_timestamp(ts):
   if not ts:
       return None
   if isinstance(ts, datetime):
       return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
   try:
       parsed = datetime.fromisoformat(ts)
       return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
   except:
       return None


async def retrieve_fn(query, extra=False):
    '''
    Same retrieval as chat endpoint, using for escalation 
    '''
    k = 10 if extra else 5

    matches = searcher.search_vectors(query, limit=k)

    matches = [
        m for m in (matches or [])
        if isinstance(m, dict)
    ]

    evidence_list = []
    for m in matches:
        url = m.get("source_url", "")
        text = m.get("text", "")
        raw_type = m.get("fact_type", "").lower()
        news_site = m.get("news_site", "")

        timestamp = m.get("timestamp") or m.get("filing_date")

        if raw_type in ["10-k", "10k", "10-q", "10q"]:
            source_type = "financial_filing"
        elif raw_type in ["news", "news_article"]:
            source_type = "news_article"
        else:
            source_type = classify_source(url, text)

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

    return evidence_list