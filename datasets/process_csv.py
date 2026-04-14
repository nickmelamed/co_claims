#!/usr/bin/env python3
"""
Process GDELT CSVs from S3, apply cleaning/company matching,
and write cleaned CSV files back to S3 under gdelt_cleaned/.
"""

from __future__ import annotations

import argparse
import json
import gzip
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse
import urllib.request

import boto3
import pandas as pd
from io import BytesIO, StringIO

try:
    from dateutil import parser as date_parser
except Exception as e:
    raise SystemExit(
        "Missing dependency 'python-dateutil'. Install with: pip install python-dateutil"
    ) from e


# ── constants ─────────────────────────────────────────────────────────────────

NEG_CUES = r"\b(decrease|decline|down|lower|pressure|headwind|soft|weak|challenging|unfavorable|adverse|constraint|shortage)\b"
POS_CUES = r"\b(increase|grow|up|higher|strong|improve|favorable|benefit|tailwind|expand|expansion|accelerate|record)\b"

SUFFIX_RE = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|ag|nv|sa|group|holdings|holding)\.?$",
    flags=re.IGNORECASE,
)

DATE_PATTERNS = [
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}",
    r"\b\d{4}-\d{2}-\d{2}",
    r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def extract_domain(url: str | None) -> str | None:
    if not url:
        return None
    return urlparse(url).netloc.replace("www.", "")


def extract_date(text: str | None):
    if text is None:
        return None
    if not isinstance(text, str):
        try:
            if pd.isna(text):
                return None
        except Exception:
            pass
        text = str(text)
    if not text:
        return None
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return date_parser.parse(match.group(), fuzzy=True)
            except Exception:
                continue
    return None


def classify_direction(s: str | None) -> str:
    if s is None:
        s_l = ""
    else:
        try:
            if pd.isna(s):
                s_l = ""
            else:
                s_l = str(s).lower()
        except Exception:
            s_l = str(s).lower()
    neg = len(re.findall(NEG_CUES, s_l))
    pos = len(re.findall(POS_CUES, s_l))
    if neg > pos and neg >= 1:
        return "negative"
    if pos > neg and pos >= 1:
        return "positive"
    return "neutral"


def norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def aliases(company_name: str):
    base = norm(company_name)
    if not base:
        return set()
    out = {base}
    while True:
        stripped = SUFFIX_RE.sub("", base).strip()
        if stripped == base or not stripped:
            break
        out.add(stripped)
        base = stripped
    return {x for x in out if len(x) >= 3}


def build_sec_lookup(user_agent: str):
    print("Fetching SEC company tickers...")
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
        if r.headers.get("Content-Encoding", "").lower() == "gzip":
            data = gzip.decompress(data)
        raw = json.loads(data.decode("utf-8"))

    sec_companies = (
        pd.DataFrame.from_dict(raw, orient="index")
        .rename(columns={"title": "company_name", "cik_str": "cik"})
    )
    sec_companies["ticker"] = sec_companies["ticker"].astype(str).str.upper().str.strip()
    sec_companies["company_name"] = sec_companies["company_name"].astype(str).str.strip()

    alias_to_rows = defaultdict(list)
    first_token_index = defaultdict(list)

    for _, r in sec_companies[["company_name", "ticker", "cik"]].dropna().iterrows():
        for a in aliases(r["company_name"]):
            alias_to_rows[a].append((r["company_name"], r["ticker"], r["cik"]))

    for a, rows in alias_to_rows.items():
        first = a.split(" ", 1)[0]
        first_token_index[first].append((a, rows))

    print(f"SEC lookup built — {len(first_token_index)} first-tokens indexed")
    return first_token_index


def extract_company_matches(title, text, first_token_index):
    combined = norm(f"{title or ''} {text or ''}")
    if not combined:
        return []
    padded = f" {combined} "
    tokens = set(combined.split())
    matches = []
    seen = set()
    for t in tokens:
        for a, rows in first_token_index.get(t, []):
            if f" {a} " in padded:
                for company_name, ticker, cik in rows:
                    key = (company_name, ticker, cik)
                    if key not in seen:
                        seen.add(key)
                        matches.append(key)
    return matches


# ── core processing ───────────────────────────────────────────────────────────

def process_df(df: pd.DataFrame, first_token_index) -> pd.DataFrame:
    if "text" not in df.columns and "article_text" in df.columns:
        df = df.rename(columns={"article_text": "text"})
    if "title" not in df.columns and "headline" in df.columns:
        df = df.rename(columns={"headline": "title"})

    if "url" not in df.columns:
        df["url"] = None
    if "title" not in df.columns:
        df["title"] = None
    if "text" not in df.columns:
        df["text"] = None

    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    if "news_site" not in df.columns:
        df["news_site"] = df["url"].apply(extract_domain)

    df["title_date"] = df["title"].apply(extract_date)
    df["text_date"] = df["text"].apply(extract_date)
    df["headline_direction"] = df["title"].apply(classify_direction)
    df["text_direction"] = df["text"].apply(classify_direction)

    df["cik"] = None
    df["accession_number"] = None
    df["section"] = "news_article"
    df["form_type"] = "news_article"
    df["fact_type"] = "news_article"
    df["direction"] = df["text_direction"]
    df["evidence_text"] = df["text"]
    df["source_url"] = df["url"]
    df["sent_index"] = None
    df["filing_date"] = df["text_date"].where(df["text_date"].notna(), df["title_date"])

    df["company_matches"] = df.apply(
        lambda r: extract_company_matches(r.get("title"), r.get("text"), first_token_index),
        axis=1,
    )

    df_long = df.explode("company_matches").copy()

    def _norm_match(x):
        if isinstance(x, (list, tuple)) and len(x) == 3:
            return x
        return (None, None, None)

    matches = df_long["company_matches"].apply(_norm_match).tolist()
    df_long[["company_name", "ticker", "cik"]] = pd.DataFrame(matches, index=df_long.index)
    df_long = df_long.drop(columns=["company_matches"])

    primary_order = [
        "cik", "ticker", "accession_number", "form_type", "filing_date",
        "section", "fact_type", "direction", "evidence_text", "source_url", "sent_index",
    ]
    front = [c for c in primary_order if c in df_long.columns]
    rest = [c for c in df_long.columns if c not in front]
    return df_long[front + rest]


# ── S3 helpers ────────────────────────────────────────────────────────────────

def list_all_csvs(s3_client, bucket: str, prefix: str) -> list[str]:
    keys = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.csv'):
                keys.append(obj['Key'])
    return keys


def already_processed(s3_client, bucket: str, out_prefix: str, stem: str) -> bool:
    key = f"{out_prefix}{stem}_cleaned.csv"
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


# ── entry points ──────────────────────────────────────────────────────────────

def process_all(bucket: str, source_prefix: str, out_prefix: str,
                user_agent: str, skip_existing: bool, csv_sep: str):
    """Used when running locally as a script."""
    s3 = boto3.client('s3')
    first_token_index = build_sec_lookup(user_agent)

    csv_keys = list_all_csvs(s3, bucket, source_prefix)
    print(f"Found {len(csv_keys)} CSVs under s3://{bucket}/{source_prefix}")

    skipped = processed = errors = 0

    for i, key in enumerate(csv_keys, 1):
        stem = Path(key).stem

        if skip_existing and already_processed(s3, bucket, out_prefix, stem):
            skipped += 1
            continue

        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(obj['Body'], sep=csv_sep, on_bad_lines='skip', low_memory=False)

            if df.empty:
                print(f"[{i}/{len(csv_keys)}] EMPTY  {key}")
                continue

            cleaned = process_df(df, first_token_index)
            relative = str(Path(key).relative_to(source_prefix.rstrip('/')))
            out_key = f"{out_prefix}{Path(relative).parent}/{stem}_cleaned.csv"
            s3.put_object(Bucket=bucket, Key=out_key, Body=cleaned.to_csv(index=False))

            processed += 1
            print(f"[{i}/{len(csv_keys)}] OK     {key} → {out_key}  rows={len(cleaned)}")

        except Exception as e:
            errors += 1
            print(f"[{i}/{len(csv_keys)}] ERROR  {key}: {e}")
            continue

    print(f"\nDone. processed={processed}  skipped={skipped}  errors={errors}")


def lambda_handler(event, context):
    """Used when deployed as a Lambda — triggered by S3 ObjectCreated."""
    import os

    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # Only process CSVs in the source prefix
    source_prefix = os.environ.get('SOURCE_PREFIX', 'gdelt/')
    out_prefix = os.environ.get('OUT_PREFIX', 'gdelt_cleaned/')
    user_agent = os.environ.get('SEC_USER_AGENT', 'Chase chasecapanna@berkeley.edu')
    csv_sep = os.environ.get('CSV_SEP', ',')

    if not key.startswith(source_prefix) or not key.endswith('.csv'):
        return {'statusCode': 200, 'body': f'Skipped {key}'}

    s3 = boto3.client('s3')
    stem = Path(key).stem

    if already_processed(s3, bucket, out_prefix, stem):
        return {'statusCode': 200, 'body': f'Already processed {key}'}

    first_token_index = build_sec_lookup(user_agent)

    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'], sep=csv_sep, on_bad_lines='skip', low_memory=False)

    if df.empty:
        return {'statusCode': 200, 'body': f'Empty file {key}'}

    cleaned = process_df(df, first_token_index)
    relative = str(Path(key).relative_to(source_prefix.rstrip('/')))
    out_key = f"{out_prefix}{Path(relative).parent}/{stem}_cleaned.csv"
    s3.put_object(Bucket=bucket, Key=out_key, Body=cleaned.to_csv(index=False))

    print(f"Processed {key} → {out_key}  rows={len(cleaned)}")
    return {'statusCode': 200, 'body': f'Processed {key}'}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket",         required=True)
    parser.add_argument("--source-prefix",  default="gdelt/")
    parser.add_argument("--out-prefix",     default="gdelt_cleaned/")
    parser.add_argument("--sec-user-agent", default="Chase chasecapanna@berkeley.edu")
    parser.add_argument("--skip-existing",  action="store_true")
    parser.add_argument("--sep",            default=",")
    args = parser.parse_args()

    process_all(
        bucket=args.bucket,
        source_prefix=args.source_prefix,
        out_prefix=args.out_prefix,
        user_agent=args.sec_user_agent,
        skip_existing=args.skip_existing,
        csv_sep=args.sep,
    )


if __name__ == "__main__":
    main()