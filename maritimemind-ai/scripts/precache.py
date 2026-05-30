#!/usr/bin/env python3
"""
scripts/precache.py
====================
Pre-cache demo queries into Redis before a live demo.

Usage:
    python scripts/precache.py [--api-url http://localhost:8000] [--token <jwt>]

What it does:
    1. Authenticates with the API to get a JWT token
    2. Sends all demo queries to /api/v1/query
    3. Results are cached in Redis (TTL: 10 minutes)
    4. Subsequent identical queries return in < 100ms from cache

Run this 5-10 minutes before a live demo.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

# ─── Default Demo Query Set ──────────────────────────────────────────────────

DEMO_QUERIES = [
    # PROCEDURE
    {
        "query": "What is the procedure for starting the main engine?",
        "category": "PROCEDURE",
        "expected_intent": "procedure",
    },
    {
        "query": "How do I perform a main engine slow turning before starting?",
        "category": "PROCEDURE",
        "expected_intent": "procedure",
    },
    {
        "query": "List the steps for auxiliary engine startup",
        "category": "PROCEDURE",
        "expected_intent": "procedure",
    },
    # EMERGENCY
    {
        "query": "Engine room flooding — immediate actions",
        "category": "EMERGENCY",
        "expected_intent": "emergency",
    },
    {
        "query": "Main engine fire emergency procedure",
        "category": "EMERGENCY",
        "expected_intent": "emergency",
    },
    # TROUBLESHOOTING
    {
        "query": "High lube oil temperature alarm on main engine — root cause",
        "category": "TROUBLESHOOTING",
        "expected_intent": "troubleshooting",
    },
    {
        "query": "Low jacket cooling water pressure — causes and remedies",
        "category": "TROUBLESHOOTING",
        "expected_intent": "troubleshooting",
    },
    # DIAGRAM REQUEST
    {
        "query": "Show me the cooling water system schematic",
        "category": "DIAGRAM",
        "expected_intent": "diagram_request",
    },
    {
        "query": "Main engine fuel oil system diagram",
        "category": "DIAGRAM",
        "expected_intent": "diagram_request",
    },
    # REGULATORY / SOP
    {
        "query": "What is MARPOL Annex VI NOx Tier III limit?",
        "category": "SOP",
        "expected_intent": "sop_lookup",
    },
    {
        "query": "SOLAS Chapter II-2 fire detection requirements",
        "category": "SOP",
        "expected_intent": "sop_lookup",
    },
    # EXPLANATION
    {
        "query": "Explain the principle of operation of a turbocharger",
        "category": "EXPLANATION",
        "expected_intent": "explanation",
    },
    {
        "query": "What is the purpose of the crosshead in a 2-stroke marine diesel engine?",
        "category": "EXPLANATION",
        "expected_intent": "explanation",
    },
]


def get_token(api_url: str, username: str, password: str) -> str:
    """Authenticate and return JWT access token."""
    resp = requests.post(
        f"{api_url}/api/v1/auth/login",
        json={"username": username, "password": password},
        timeout=10,
    )
    if resp.status_code != 200:
        print(f"  ✗ Auth failed: {resp.status_code} {resp.text}")
        sys.exit(1)
    token = resp.json().get("access_token")
    print(f"  ✓ Authenticated (token: {token[:20]}...)")
    return token


def precache_query(api_url: str, query: str, token: str, category: str) -> dict:
    """Send a single query to the API to trigger caching."""
    resp = requests.post(
        f"{api_url}/api/v1/query",
        json={"query": query},
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,  # LLM can be slow
    )
    if resp.status_code == 200:
        data = resp.json()
        confidence = data.get("confidence", 0)
        intent = data.get("intent", "unknown")
        n_citations = len(data.get("citations", []))
        n_images = len(data.get("images", []))
        return {
            "status": "ok",
            "confidence": confidence,
            "intent": intent,
            "citations": n_citations,
            "images": n_images,
        }
    else:
        return {"status": "error", "code": resp.status_code, "detail": resp.text[:100]}


def main():
    parser = argparse.ArgumentParser(description="Pre-cache demo queries for MaritimeMind AI")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--username", default="admin", help="Login username")
    parser.add_argument("--password", default="password", help="Login password")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between queries (seconds)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  MaritimeMind AI — Demo Pre-Cache Script")
    print(f"  API: {args.api_url}")
    print(f"{'='*60}\n")

    # Authenticate
    print("► Authenticating...")
    token = get_token(args.api_url, args.username, args.password)

    # Pre-cache each query
    print(f"\n► Caching {len(DEMO_QUERIES)} demo queries...\n")
    results = []

    for i, item in enumerate(DEMO_QUERIES, 1):
        query = item["query"]
        category = item["category"]
        print(f"  [{i:02d}/{len(DEMO_QUERIES)}] [{category}] {query[:60]}...")

        start = time.time()
        result = precache_query(args.api_url, query, token, category)
        elapsed = time.time() - start

        if result["status"] == "ok":
            print(
                f"       ✓ {elapsed:.1f}s | conf={result['confidence']:.2f} | "
                f"intent={result['intent']} | citations={result['citations']} | "
                f"images={result['images']}"
            )
        else:
            print(f"       ✗ Error {result['code']}: {result['detail']}")

        results.append({**item, **result, "elapsed_s": round(elapsed, 2)})

        if i < len(DEMO_QUERIES):
            time.sleep(args.delay)

    # Summary
    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\n{'='*60}")
    print(f"  ✓ Cached: {ok}/{len(DEMO_QUERIES)} queries")
    print(f"  Next identical queries will return from Redis cache (<100ms)")
    print(f"  Cache TTL: 10 minutes")
    print(f"{'='*60}\n")

    # Save results
    out_path = Path("data/demo/precache_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
