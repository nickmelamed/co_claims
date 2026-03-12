#!/usr/bin/env python3
"""
Download the news_media_reliability dataset from Hugging Face,
write to parquet, and optionally zip it.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

try:
    from datasets import load_dataset
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'datasets'. Install with: pip install datasets"
    ) from e


def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--out",
        default=str(script_dir / "news_media_reliability.parquet"),
        help="Output parquet path",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also create a .zip alongside the parquet",
    )
    args = parser.parse_args()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("sergioburdisso/news_media_reliability", split="train")
    df = pd.DataFrame(ds)
    df.to_parquet(out_path, index=False)
    print(f"wrote {out_path} rows={len(df)}")

    if args.zip:
        zip_path = out_path.with_suffix(out_path.suffix + ".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path, arcname=out_path.name)
        print(f"wrote {zip_path}")


if __name__ == "__main__":
    main()
