"""Extract stock tickers from company names mentioned in text fields.

This module is offline-only: provide a local company mapping dataframe with
columns like "company_name" and "ticker".

Usage (in notebooks):

    from co_claims.ticker_extraction import annotate_dataframe_with_tickers

    df = annotate_dataframe_with_tickers(
        df,
        companies_df=tech_companies,
        title_col="title",
        text_col="text",
        company_name_col="company_name",
        ticker_col="ticker",
        output_col="tickers",
    )
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

_SUFFIX_PATTERN = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|ag|nv|sa|group|holdings|holding)\.?$",
    flags=re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _name_aliases(company_name: str) -> Set[str]:
    normalized = _normalize_text(company_name)
    if not normalized:
        return set()

    aliases = {normalized}
    base = normalized

    # Strip legal suffixes to match shorter mentions in article text.
    while True:
        stripped = _SUFFIX_PATTERN.sub("", base).strip()
        if stripped == base or not stripped:
            break
        aliases.add(stripped)
        base = stripped

    return {a for a in aliases if len(a) >= 3}


def build_match_index(
    companies_df: pd.DataFrame,
    company_name_col: str = "company_name",
    ticker_col: str = "ticker",
) -> Dict[str, List[Tuple[str, Set[str]]]]:
    """Build a fast first-token index from a local company->ticker dataframe."""
    if company_name_col not in companies_df.columns or ticker_col not in companies_df.columns:
        missing = [c for c in (company_name_col, ticker_col) if c not in companies_df.columns]
        raise KeyError(f"Missing required columns in companies_df: {missing}")

    alias_to_symbols: Dict[str, Set[str]] = defaultdict(set)

    records: Iterable[dict] = companies_df[[company_name_col, ticker_col]].to_dict("records")
    for record in records:
        company_name = str(record.get(company_name_col, "") or "").strip()
        symbol = str(record.get(ticker_col, "") or "").strip().upper()
        if not company_name or not symbol:
            continue

        for alias in _name_aliases(company_name):
            alias_to_symbols[alias].add(symbol)

    first_token_index: Dict[str, List[Tuple[str, Set[str]]]] = defaultdict(list)
    for alias, symbols in alias_to_symbols.items():
        first_token = alias.split(" ", 1)[0]
        first_token_index[first_token].append((alias, symbols))

    return dict(first_token_index)


def extract_tickers_from_text(
    title: str,
    text: str,
    match_index: Dict[str, List[Tuple[str, Set[str]]]],
) -> List[str]:
    combined = _normalize_text(f"{title or ''} {text or ''}")
    if not combined:
        return []

    padded = f" {combined} "
    tokens = set(combined.split())
    matches: Set[str] = set()

    for token in tokens:
        for alias, symbols in match_index.get(token, []):
            if f" {alias} " in padded:
                matches.update(symbols)

    return sorted(matches)


def annotate_dataframe_with_tickers(
    df: pd.DataFrame,
    companies_df: pd.DataFrame,
    title_col: str = "title",
    text_col: str = "text",
    company_name_col: str = "company_name",
    ticker_col: str = "ticker",
    output_col: str = "tickers",
) -> pd.DataFrame:
    """Annotate `df` with a ticker list based on mentions in title/text."""
    if title_col not in df.columns or text_col not in df.columns:
        missing = [c for c in (title_col, text_col) if c not in df.columns]
        raise KeyError(f"Missing required columns in df: {missing}")

    match_index = build_match_index(
        companies_df=companies_df,
        company_name_col=company_name_col,
        ticker_col=ticker_col,
    )

    out = df.copy()
    out[output_col] = out.apply(
        lambda row: extract_tickers_from_text(
            str(row.get(title_col, "") or ""),
            str(row.get(text_col, "") or ""),
            match_index=match_index,
        ),
        axis=1,
    )
    return out
