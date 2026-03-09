"""
Overnight GDELT article scraper for multiple financial news sources.
Async concurrent scraping for speed. ~10-20 articles at once.

Usage:
    python scraper.py
    python scraper.py --start 2024-01 --end 2026-03
    python scraper.py --resume
"""

import asyncio
import aiohttp
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse
import logging
import time
import json
import sys
import re
import html
import gzip
import zlib
from urllib.parse import quote, urlparse
try:
    import brotli  # type: ignore
except Exception:
    try:
        import brotlicffi as brotli  # type: ignore
    except Exception:
        brotli = None

# --- Config ---
SOURCES = [
    "cnbc.com",
    "reuters.com",
    "finance.yahoo.com",
    "nasdaq.com",
    "cnn.com",
]
MAX_RECORDS = 250
CONCURRENCY = 20       # simultaneous article scrapes
OUTPUT_DIR = Path("scraped_articles")
PROGRESS_FILE = OUTPUT_DIR / "progress.json"
GDELT_DELAY = 6
MAX_RETRIES = 5
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DOMAIN_QUERIES = {
    "nasdaq.com": "(domain:nasdaq.com OR domain:www.nasdaq.com)",
    "cnn.com": "(domain:cnn.com OR domain:www.cnn.com)",
}
BLOCKED_PAGE_MARKERS = (
    "please enable js and disable any ad blocker",
    "enable javascript",
    "are you a robot",
    "access denied",
    "reuters.com ===============",
    "we're sorry, but the content you're looking for is no longer available",
    "engage with, participate in, and build your own modern markets",
)
MIN_ARTICLE_CHARS = 120

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# --- Selectors per site ---
SELECTORS_BY_DOMAIN = {
    "cnbc.com": ["div.ArticleBody-articleBody", "[data-module='ArticleBody']"],
    "reuters.com": ["div.article-body__content", "div[data-testid='paragraph']"],
    "finance.yahoo.com": [
        "div.caas-body",
        "article div.caas-body",
        "article [data-testid='article-content-wrapper']",
    ],
    "nasdaq.com": ["article", "[class*='article-body']", "[class*='article__content']"],
    "cnn.com": ["article", ".article__content", ".article-body", "[data-editable='content']"],
}
FALLBACK_SELECTORS = ["article", "main", "body"]


# --- Helpers ---

def week_ranges(start: str, end: str):
    current = datetime.strptime(start, "%Y-%m")
    end_dt = datetime.strptime(end, "%Y-%m") + relativedelta(months=1) - relativedelta(seconds=1)
    while current <= end_dt:
        week_end = min(current + relativedelta(days=6, hours=23, minutes=59, seconds=59), end_dt)
        yield (
            current.strftime("%Y%m%d%H%M%S"),
            week_end.strftime("%Y%m%d%H%M%S"),
            current.strftime("%Y-%m-%d"),
        )
        current = week_end + relativedelta(seconds=1)


def gdelt_request(domain: str, start_dt: str, end_dt: str) -> list[dict] | None:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    query = DOMAIN_QUERIES.get(domain, f"domain:{domain}")
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": MAX_RECORDS,
        "startdatetime": start_dt,
        "enddatetime": end_dt,
        "format": "json",
    }
    saw_rate_limit = False
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200 and "json" in resp.headers.get("Content-Type", ""):
                data = resp.json()
                return data.get("articles", [])
            if resp.status_code == 429:
                saw_rate_limit = True
                wait = max(GDELT_DELAY * (2 ** attempt), 30)
            else:
                wait = GDELT_DELAY * (2 ** attempt)
            log.warning(f"GDELT {resp.status_code} for {domain} ({start_dt[:6]}). Retry in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            wait = GDELT_DELAY * (2 ** attempt)
            log.error(f"GDELT error: {e}. Retry in {wait}s...")
            time.sleep(wait)
    log.error(f"Failed after {MAX_RETRIES} retries: {domain} {start_dt[:6]}")
    if saw_rate_limit:
        return None
    return []


# --- Async scraping ---

def _extract_paragraph_text(node: BeautifulSoup) -> str | None:
    paragraphs = [p.get_text(" ", strip=True) for p in node.find_all("p")]
    text = " ".join(p for p in paragraphs if p)
    return text or None


def _from_caas_art_html(raw_html: str) -> str | None:
    soup = BeautifulSoup(raw_html, "html.parser")
    return _extract_paragraph_text(soup)


def _json_find_text(obj) -> str | None:
    if isinstance(obj, dict):
        for key in ("articleBody", "description"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        for key in ("caasArtHtml", "content"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                if "<" in value and ">" in value:
                    text = _from_caas_art_html(value)
                    if text:
                        return text
                if len(value.strip()) >= MIN_ARTICLE_CHARS:
                    return value.strip()

        for value in obj.values():
            found = _json_find_text(value)
            if found:
                return found

    if isinstance(obj, list):
        for item in obj:
            found = _json_find_text(item)
            if found:
                return found
    return None


def _extract_ld_json_text(soup: BeautifulSoup) -> str | None:
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        text = _json_find_text(payload)
        if text:
            return text
    return None


def _extract_root_app_text(soup: BeautifulSoup) -> str | None:
    pattern = re.compile(r"root\.App\.main\s*=\s*(\{.*\})\s*;\s*$", re.DOTALL)
    for script in soup.find_all("script"):
        raw = script.string or script.get_text()
        if not raw or "root.App.main" not in raw:
            continue
        match = pattern.search(raw)
        if not match:
            continue
        blob = match.group(1)
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            continue
        text = _json_find_text(payload)
        if text:
            return text
    return None


def parse_article(html_bytes: bytes, url: str) -> str | None:
    soup = BeautifulSoup(html_bytes, "html.parser")
    host = urlparse(url).netloc.lower()

    for domain, selectors in SELECTORS_BY_DOMAIN.items():
        if domain in host:
            for selector in selectors:
                body = soup.select_one(selector)
                if body:
                    text = _extract_paragraph_text(body)
                    if text:
                        return text

    for selector in FALLBACK_SELECTORS:
        body = soup.select_one(selector)
        if body:
            text = _extract_paragraph_text(body)
            if text:
                return text

    text = _extract_ld_json_text(soup)
    if text:
        return html.unescape(text)

    text = _extract_root_app_text(soup)
    if text:
        return html.unescape(text)

    paragraphs = soup.find_all("p")
    if paragraphs:
        return " ".join(p.get_text(" ", strip=True) for p in paragraphs)
    return None


def is_blocked_text(text: str | None) -> bool:
    if not text:
        return True
    normalized = text.lower()
    return any(marker in normalized for marker in BLOCKED_PAGE_MARKERS)


def looks_like_article(text: str | None) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < MIN_ARTICLE_CHARS or is_blocked_text(stripped):
        return False

    # Guardrail against compressed/binary garbage accidentally decoded as text.
    replacement_ratio = stripped.count("�") / max(len(stripped), 1)
    non_printable = sum(1 for ch in stripped if (not ch.isprintable()) and ch not in "\n\r\t")
    non_printable_ratio = non_printable / max(len(stripped), 1)
    if replacement_ratio > 0.02 or non_printable_ratio > 0.01:
        return False
    return True


def looks_english_enough(text: str | None, min_ascii_ratio: float = 0.75) -> bool:
    if not text:
        return False
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    ascii_letters = sum(1 for ch in letters if "a" <= ch.lower() <= "z")
    return (ascii_letters / len(letters)) >= min_ascii_ratio


def clean_reader_text(text: str) -> str | None:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    for marker in ("URL Source:", "Title:", "Published Time:", "Markdown Content:"):
        cleaned = cleaned.replace(marker, f"\n{marker}")
    if "Markdown Content:" in cleaned:
        cleaned = cleaned.split("Markdown Content:", 1)[1]
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines() if line.strip())
    if not looks_like_article(cleaned):
        return None
    return cleaned


async def fetch_via_reader(session: aiohttp.ClientSession, url: str) -> str | None:
    # r.jina.ai often returns readable text for JS-heavy pages that block plain HTML scraping.
    reader_url = f"https://r.jina.ai/{quote(url, safe=':/?&=%')}"
    try:
        async with session.get(reader_url, timeout=aiohttp.ClientTimeout(total=25)) as resp:
            if resp.status != 200:
                return None
            body = await resp.text()
            return clean_reader_text(body)
    except Exception:
        return None


def decode_response_body(raw: bytes, content_encoding: str | None) -> bytes:
    if not raw:
        return raw
    encoding = (content_encoding or "").lower()
    try:
        if "br" in encoding and brotli is not None:
            return brotli.decompress(raw)
        if "gzip" in encoding:
            return gzip.decompress(raw)
        if "deflate" in encoding:
            try:
                return zlib.decompress(raw)
            except zlib.error:
                return zlib.decompress(raw, -zlib.MAX_WBITS)
        # Fallback for mislabelled responses that are actually gzip.
        if raw.startswith(b"\x1f\x8b"):
            return gzip.decompress(raw)
    except Exception:
        # Some endpoints send invalid/mislabelled compression; keep raw bytes as fallback.
        return raw
    return raw


async def fetch_article(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    url: str,
    alt_url: str | None = None,
) -> str | None:
    async with sem:
        attempts = [url]
        if alt_url and alt_url != url:
            attempts.append(alt_url)

        for candidate_url in attempts:
            try:
                async with session.get(candidate_url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    raw = await resp.read()
                    body = decode_response_body(raw, resp.headers.get("Content-Encoding"))
                    parsed = parse_article(body, candidate_url)
                    host = urlparse(candidate_url).netloc.lower()
                    if looks_like_article(parsed):
                        return parsed
                    if "reuters.com" in host or "finance.yahoo.com" in host or "nasdaq.com" in host or "cnn.com" in host:
                        fallback = await fetch_via_reader(session, candidate_url)
                        if looks_like_article(fallback):
                            return fallback
                    if "reuters.com" in host and is_blocked_text(parsed):
                        continue
                    if looks_like_article(parsed):
                        return parsed
            except Exception as e:
                log.warning(f"Scrape failed: {candidate_url} ({e})")
        return None


async def scrape_batch(urls: list[str], alt_urls: list[str | None] | None = None) -> list[str | None]:
    sem = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False)
    async with aiohttp.ClientSession(
        connector=connector,
        auto_decompress=False,
        max_line_size=32768,
        max_field_size=32768,
    ) as session:
        if alt_urls is None:
            alt_urls = [None] * len(urls)
        tasks = [fetch_article(session, sem, url, alt_url) for url, alt_url in zip(urls, alt_urls)]
        results = await asyncio.gather(*tasks)
    return list(results)


# --- Progress tracking ---

def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {}

def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))

def is_done(progress: dict, domain: str, label: str):
    return progress.get(domain, {}).get(label, False)

def mark_done(progress: dict, domain: str, label: str):
    progress.setdefault(domain, {})[label] = True
    save_progress(progress)


# --- Main ---

def run(start: str, end: str, resume: bool):
    OUTPUT_DIR.mkdir(exist_ok=True)

    fh = logging.FileHandler(OUTPUT_DIR / "scraper.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    progress = load_progress() if resume else {}

    for domain in SOURCES:
        log.info(f"=== Starting {domain} ===")
        domain_dir = OUTPUT_DIR / domain.replace(".", "_")
        domain_dir.mkdir(exist_ok=True)

        for start_dt, end_dt, week_label in week_ranges(start, end):
            if is_done(progress, domain, week_label):
                log.info(f"Skipping {domain} {week_label} (done)")
                continue

            log.info(f"Fetching {domain} week of {week_label}...")
            articles = gdelt_request(domain, start_dt, end_dt)
            if articles is None:
                log.warning(f"Rate-limited on {domain} {week_label}; leaving week unmarked for retry.")
                time.sleep(GDELT_DELAY * 2)
                continue

            if not articles:
                log.info(f"No articles for {domain} {week_label}")
                mark_done(progress, domain, week_label)
                time.sleep(GDELT_DELAY)
                continue

            df = pd.DataFrame(articles)
            if domain == "reuters.com" and "language" in df.columns:
                before = len(df)
                df = df[df["language"].fillna("").str.lower() == "english"].copy()
                removed = before - len(df)
                if removed:
                    log.info(f"Filtered Reuters non-English rows by metadata: removed {removed}, kept {len(df)}")
                if df.empty:
                    log.info(f"No English Reuters rows for {week_label} after filtering")
                    mark_done(progress, domain, week_label)
                    time.sleep(GDELT_DELAY)
                    continue
            urls = df["url"].tolist()
            alt_urls: list[str | None] = [None] * len(urls)
            if domain == "cnn.com" and "url_mobile" in df.columns:
                mobile_urls = df["url_mobile"].fillna("").astype(str).tolist()
                desktop_urls = df["url"].fillna("").astype(str).tolist()
                urls = [m if m else d for m, d in zip(mobile_urls, desktop_urls)]
                alt_urls = [d if m and d and m != d else None for m, d in zip(mobile_urls, desktop_urls)]
            log.info(f"Got {len(urls)} URLs. Scraping concurrently ({CONCURRENCY} at a time)...")

            texts = asyncio.run(scrape_batch(urls, alt_urls))
            if domain == "reuters.com":
                texts = [t if looks_english_enough(t) else None for t in texts]
            df["article_text"] = texts

            out_file = domain_dir / f"{week_label}.csv"
            df.to_csv(out_file, index=False)

            success = sum(1 for t in texts if t)
            log.info(f"Saved {out_file} — {success}/{len(df)} with text")

            mark_done(progress, domain, week_label)
            time.sleep(GDELT_DELAY)

    log.info("=== All done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDELT financial news scraper")
    parser.add_argument("--start", default="2020-01", help="Start month (YYYY-MM)")
    parser.add_argument("--end", default="2026-03", help="End month (YYYY-MM)")
    parser.add_argument("--resume", action="store_true", help="Resume from last progress")
    args = parser.parse_args()
    run(args.start, args.end, args.resume)
