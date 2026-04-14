#!/usr/bin/env python3
"""
Filter cleaned parquet files by ticker/company name and export to parquet+csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TICKERS = [
    "AAPL", "ADBE", "AMD", "AMZN", "AVGO", "CRM", "CSCO", "GOOG", "META",
    "MSFT", "NFLX", "NOW", "NVDA", "ORCL", "QCOM", "SHOP", "TSLA",
]

# Tickers that are common English words and cause false positives in text.
TEXT_AMBIGUOUS_TICKERS = {"NOW", "SHOP"}

COMPANY_NAMES = {
    "Apple Inc.",
    "Adobe Inc.",
    "Advanced Micro Devices, Inc.",
    "Amazon.com, Inc.",
    "Broadcom Inc.",
    "Salesforce, Inc.",
    "Cisco Systems, Inc.",
    "Alphabet Inc.",
    "Meta Platforms, Inc.",
    "Microsoft Corporation",
    "Netflix, Inc.",
    "ServiceNow, Inc.",
    "NVIDIA Corporation",
    "Oracle Corporation",
    "QUALCOMM Incorporated",
    "Shopify Inc.",
    "Tesla, Inc.",
}


def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input-dir",
        default=str(script_dir / "parquet_by_domain"),
        help="Directory containing *_cleaned.parquet files",
    )
    parser.add_argument(
        "--out-base",
        default=str(script_dir / "filtered_tickers"),
        help="Output base path without extension",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.1,
        help="Deterministic sample fraction in (0,1]; default 0.1",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed for deterministic sampling hash key",
    )
    parser.add_argument(
        "--dedupe-text",
        action="store_true",
        default=True,
        help="Drop records with duplicate text/evidence_text (default: on)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    files = sorted(input_dir.glob("*_cleaned.parquet"))
    if not files:
        raise SystemExit(f"No cleaned parquet files found in {input_dir}")

    if not (0 < args.sample_frac <= 1.0):
        raise SystemExit("--sample-frac must be in (0, 1]")

    out_parquet = Path(f"{args.out_base}.parquet")
    out_csv = Path(f"{args.out_base}.csv")
    # Case-sensitive match to avoid false positives like "now" matching "NOW".
    text_tickers = [t for t in TICKERS if t not in TEXT_AMBIGUOUS_TICKERS]
    ticker_pattern = r"(?<![A-Z0-9])(?:\\$)?(?:" + "|".join(text_tickers) + r")(?![A-Z0-9])"
    hash_key = f"{args.sample_seed:016d}"

    # Ensure we overwrite prior outputs rather than append.
    if out_parquet.exists():
        out_parquet.unlink()
    if out_csv.exists():
        out_csv.unlink()

    # Unify schemas across files to avoid writer mismatches (e.g., null vs timestamp).
    schemas = [pq.ParquetFile(f).schema_arrow for f in files]
    target_schema = pa.unify_schemas(schemas)

    writer: pq.ParquetWriter | None = None
    wrote_csv = False
    total_rows = 0
    seen_text_hashes: set[int] = set()

    try:
        for f in files:
            parquet_file = pq.ParquetFile(f)
            for batch in parquet_file.iter_batches():
                df = batch.to_pandas()
                # Ensure all target columns exist and coerce key types to match schema.
                for field in target_schema:
                    if field.name not in df.columns:
                        df[field.name] = pd.NA
                    if pa.types.is_timestamp(field.type):
                        df[field.name] = pd.to_datetime(df[field.name], errors="coerce")
                    elif pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                        df[field.name] = df[field.name].astype("string")
                tickers = df.get("ticker", pd.Series([""] * len(df))).fillna("").astype(str).str.upper()
                names = df.get("company_name", pd.Series([""] * len(df))).fillna("").astype(str)
                evidence = df.get("evidence_text", pd.Series([""] * len(df))).fillna("").astype(str)

                text_match = evidence.str.contains(ticker_pattern, case=True, regex=True)
                mask = tickers.isin(TICKERS) | names.isin(COMPANY_NAMES) | text_match
                filtered = df[mask]
                if filtered.empty:
                    continue

                if args.sample_frac < 1.0:
                    h = pd.util.hash_pandas_object(
                        filtered, index=False, hash_key=hash_key
                    ).astype("uint64")
                    threshold = int(args.sample_frac * (2**64 - 1))
                    filtered = filtered[h <= threshold]
                    if filtered.empty:
                        continue

                if args.dedupe_text:
                    text_col = "text" if "text" in filtered.columns else "evidence_text"
                    text_series = (
                        filtered.get(text_col, pd.Series([""] * len(filtered)))
                        .fillna("")
                        .astype(str)
                    )
                    text_hashes = pd.util.hash_pandas_object(
                        text_series, index=False, hash_key=hash_key
                    ).astype("uint64")
                    keep_mask = ~text_hashes.isin(seen_text_hashes)
                    if keep_mask.any():
                        seen_text_hashes.update(text_hashes[keep_mask].to_numpy().tolist())
                        filtered = filtered[keep_mask]
                    else:
                        continue

                total_rows += len(filtered)

                table = pa.Table.from_pandas(filtered, schema=target_schema, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_parquet, target_schema)
                writer.write_table(table)

                filtered.to_csv(out_csv, mode="a", header=not wrote_csv, index=False)
                wrote_csv = True
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        empty_table = pa.Table.from_arrays(
            [pa.array([], type=field.type) for field in target_schema],
            schema=target_schema,
        )
        pq.write_table(empty_table, out_parquet)
        pd.DataFrame(columns=[field.name for field in target_schema]).to_csv(out_csv, index=False)

    print(f"wrote {out_parquet} rows={total_rows}")
    print(f"wrote {out_csv} rows={total_rows}")


if __name__ == "__main__":
    main()
