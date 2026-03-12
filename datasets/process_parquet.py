#!/usr/bin/env python3
"""
Process zipped parquet files using the same steps as ds_eda.ipynb (pre-marker).
Creates per-domain cleaned parquet files with "_cleaned" suffix.
"""

from __future__ import annotations

import argparse
import json
import gzip
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse
import urllib.request

import pandas as pd

try:
    from dateutil import parser as date_parser
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'python-dateutil'. Install with: pip install python-dateutil"
    ) from e


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

    return first_token_index


def extract_company_matches(title: str | None, text: str | None, first_token_index):
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


def process_df(df: pd.DataFrame, first_token_index):
    # Map source columns to expected ones
    if "text" not in df.columns and "article_text" in df.columns:
        df = df.rename(columns={"article_text": "text"})

    if "title" not in df.columns and "headline" in df.columns:
        df = df.rename(columns={"headline": "title"})

    # Ensure required columns exist
    if "url" not in df.columns:
        df["url"] = None
    if "title" not in df.columns:
        df["title"] = None
    if "text" not in df.columns:
        df["text"] = None
    # Normalize to string to avoid Arrow dtype issues during apply
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    # Add news_site
    if "news_site" not in df.columns:
        df["news_site"] = df["url"].apply(extract_domain)

    # Add extracted dates
    df["title_date"] = df["title"].apply(extract_date)
    df["text_date"] = df["text"].apply(extract_date)

    # Sentiment/direction tags
    df["headline_direction"] = df["title"].apply(classify_direction)
    df["text_direction"] = df["text"].apply(classify_direction)

    # Schema alignment
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

    # Company matching
    df["company_matches"] = df.apply(
        lambda r: extract_company_matches(r.get("title"), r.get("text"), first_token_index),
        axis=1,
    )

    # Explode matches
    df_long = df.explode("company_matches").copy()
    def _norm_match(x):
        if isinstance(x, (list, tuple)) and len(x) == 3:
            return x
        return (None, None, None)

    matches = df_long["company_matches"].apply(_norm_match).tolist()
    df_long[["company_name", "ticker", "cik"]] = pd.DataFrame(
        matches, index=df_long.index
    )
    df_long = df_long.drop(columns=["company_matches"])

    # Reorder columns
    primary_order = [
        "cik",
        "ticker",
        "accession_number",
        "form_type",
        "filing_date",
        "section",
        "fact_type",
        "direction",
        "evidence_text",
        "source_url",
        "sent_index",
    ]
    front = [c for c in primary_order if c in df_long.columns]
    rest = [c for c in df_long.columns if c not in front]
    df_long = df_long[front + rest]

    return df_long


def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--zip",
        default=str(script_dir / "parquet_by_domain.zip"),
        help="Path to zip with parquet files",
    )
    parser.add_argument(
        "--out-dir",
        default=str(script_dir / "parquet_by_domain"),
        help="Output directory for cleaned parquet files",
    )
    parser.add_argument(
        "--sec-user-agent",
        default="Chase chasecapanna@berkeley.edu",
        help="User-Agent for SEC request",
    )
    args = parser.parse_args()

    zip_path = Path(args.zip).expanduser().resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    first_token_index = build_sec_lookup(args.sec_user_agent)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        shutil.unpack_archive(str(zip_path), str(tmpdir))
        parquet_files = sorted(tmpdir.glob("*.parquet"))

        if not parquet_files:
            raise SystemExit(f"No parquet files found in {zip_path}")

        for p in parquet_files:
            df = pd.read_parquet(p)
            cleaned = process_df(df, first_token_index)
            out_file = out_dir / f"{p.stem}_cleaned.parquet"
            cleaned.to_parquet(out_file, index=False)
            print(f"wrote {out_file} rows={len(cleaned)}")


if __name__ == "__main__":
    main()
