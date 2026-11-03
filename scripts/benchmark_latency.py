"""Measure end-to-end `/query` latency against a running TrustGuard server.

Requires a server with ANTHROPIC_API_KEY configured (for example
`docker compose up`). Each question is sent once, sequentially.

Usage:
    python scripts/benchmark_latency.py [--url http://localhost:8000] [--target-ms 3000]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request

QUESTIONS = [
    "What is the daily ATM withdrawal limit for a consumer debit card?",
    "When does a wire transfer need dual authorization?",
    "What is the deadline to file a SAR after initial detection?",
    "Customer Maria Garcia, card 4111 1111 1111 1111, made a $2,000 purchase in euros. "
    "What foreign transaction fee applies?",
    "What is the cash advance fee on a $150 advance?",
    "How long are account records kept after an account is closed?",
    "SSN 123-45-6789 reported a stolen debit card on day 1. What is their maximum liability?",
    "Can a generative AI assistant make final credit decisions?",
    "What is 3% of $4,250?",
    "What does the policy say about cryptocurrency custody?",
]


def ask(url: str, question: str) -> tuple[float, dict[str, object]]:
    body = json.dumps({"query": question}).encode()
    request = urllib.request.Request(
        f"{url}/query", data=body, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        payload = json.loads(response.read())
    return (time.perf_counter() - start) * 1000, payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--target-ms", type=float, default=3000)
    parser.add_argument("--verbose", action="store_true", help="Print each answer")
    args = parser.parse_args()

    latencies = []
    for question in QUESTIONS:
        elapsed, payload = ask(args.url, question)
        latencies.append(elapsed)
        tools = ",".join(payload["tools_used"]) or "-"  # type: ignore[arg-type]
        print(f"{elapsed:7.0f} ms  tools={tools:32s} {question[:60]}")
        if args.verbose:
            print(f"           masked: {payload['masked_query']}")
            print(f"           answer: {payload['answer']}")
            print(f"           cites : {payload['citations']}\n")

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[max(0, round(0.95 * len(latencies)) - 1)]
    within = sum(lat <= args.target_ms for lat in latencies)
    print(f"\np50 {p50:.0f} ms | p95 {p95:.0f} ms | max {max(latencies):.0f} ms")
    print(f"{within}/{len(latencies)} queries within {args.target_ms:.0f} ms")
    return 0 if p95 <= args.target_ms else 1


if __name__ == "__main__":
    sys.exit(main())
