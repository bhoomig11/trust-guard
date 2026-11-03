"""Measure retrieval relevance against the labeled question set.

A question counts as a hit when at least one of the top-k retrieved chunks
comes from the expected document and contains the expected phrase. Questions
are passed through the privacy shield first, exactly as in production.

Usage:
    python scripts/evaluate_retrieval.py [--k 4] [--embedding-model NAME] [--no-mask]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from trustguard.config import get_settings
from trustguard.etl.pipeline import ingest_directory
from trustguard.privacy.shield import PrivacyShield
from trustguard.vectorstore import PolicyVectorStore, SentenceTransformerEmbedder


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", type=Path, default=Path("eval/retrieval_eval.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--k", type=int, default=settings.top_k)
    parser.add_argument("--embedding-model", default=settings.embedding_model)
    parser.add_argument("--no-mask", action="store_true", help="Skip PII masking of questions")
    parser.add_argument("--show-misses", action="store_true")
    parser.add_argument("--target", type=float, default=0.90)
    args = parser.parse_args()

    store = PolicyVectorStore(SentenceTransformerEmbedder(args.embedding_model))
    report = ingest_directory(
        args.data_dir, store, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    shield = None if args.no_mask else PrivacyShield(settings.spacy_model)

    indexed = set(store.sources())
    items = [
        json.loads(line)
        for line in args.eval_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    skipped = [item for item in items if item["source"] not in indexed]
    if skipped:
        names = sorted({item["source"] for item in skipped})
        print(f"Skipping {len(skipped)} questions for documents not indexed: {', '.join(names)}")

    rows = []
    for item in items:
        if item["source"] not in indexed:
            continue
        question = shield.mask(item["question"]).masked_text if shield else item["question"]
        start = time.perf_counter()
        hits = store.search(question, args.k)
        latency_ms = (time.perf_counter() - start) * 1000
        rank = next(
            (
                i
                for i, h in enumerate(hits, start=1)
                if h.source == item["source"] and item["expect"].lower() in h.text.lower()
            ),
            None,
        )
        rows.append(
            {
                "question": item["question"],
                "masked": question,
                "source": item["source"],
                "hit": rank is not None,
                "rank": rank,
                "top_source": hits[0].source if hits else None,
                "top_score": hits[0].score if hits else None,
                "latency_ms": latency_ms,
            }
        )

    frame = pd.DataFrame(rows)
    hit_rate = float(frame["hit"].mean())
    ranked = frame.dropna(subset=["rank"])
    mrr = float((1 / ranked["rank"]).sum() / len(frame))
    print(f"Embedding model : {args.embedding_model}")
    print(f"Indexed         : {report.documents} documents, {report.chunks} chunks")
    print(f"Questions       : {len(frame)}")
    print(f"Hit@{args.k}           : {hit_rate:.1%}")
    print(f"Hit@1           : {float((frame['rank'] == 1).mean()):.1%}")
    print(f"MRR@{args.k}           : {mrr:.3f}")
    print(f"Search p95 (ms) : {frame['latency_ms'].quantile(0.95):.1f}")
    print("\nHit rate by document:")
    print(frame.groupby("source")["hit"].mean().map("{:.0%}".format).to_string())
    if args.show_misses:
        misses = frame[~frame["hit"]]
        if not misses.empty:
            print("\nMisses:")
            for row in misses.itertuples():
                print(f"  - {row.question}  (top: {row.top_source})")
    return 0 if hit_rate >= args.target else 1


if __name__ == "__main__":
    sys.exit(main())
