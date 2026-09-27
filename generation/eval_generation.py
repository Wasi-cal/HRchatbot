#!/usr/bin/env python3
"""Runs a set of test queries through the full retrieve -> generate
pipeline and prints the final answer plus the timing breakdown for each.

Reuses vectorstore/eval_retrieval.py's query set (paraphrase + exact-term
style, all grounded in the real corpus) plus deliberately out-of-corpus
queries to confirm the low-confidence shortcut actually triggers.

    python3 -m generation.eval_generation
"""
from generation.generate import generate_answer
from vectorstore.eval_retrieval import EXACT_TERM_QUERIES, PARAPHRASE_QUERIES

# Nothing in this HR policy corpus (see vectorstore/eval_retrieval.py's
# grounding check against out_chunks/*.json) could plausibly answer
# these - they exist to confirm the confidence-threshold shortcut fires
# instead of paying for a generation call.
OUT_OF_CORPUS_QUERIES = [
    "what is the weather forecast for tomorrow",
    "what is the capital of France",
    "can you recommend a good recipe for dinner",
]


def run_query(query: str):
    print(f"\n=== Query: {query!r} ===")
    answer = None
    timing = None
    shortcut = None
    for event in generate_answer(query):
        if event["type"] == "done":
            answer = event["answer"]
            timing = event["timing"]
            shortcut = event["shortcut"]

    print(f"Answer: {answer}")
    print(f"Shortcut taken (low-confidence, no generation call): {shortcut}")
    if shortcut:
        print(f"  retrieval_time_ms:        {timing['retrieval_time_ms']:.1f}")
    else:
        print(f"  retrieval_time_ms:        {timing['retrieval_time_ms']:.1f}")
        ttft = timing["time_to_first_token_ms"]
        print(f"  time_to_first_token_ms:   {ttft:.1f}" if ttft is not None else "  time_to_first_token_ms:   n/a")
        print(f"  total_generation_time_ms: {timing['total_generation_time_ms']:.1f}")


def main():
    print("############ Paraphrase-style queries ############")
    for query in PARAPHRASE_QUERIES:
        run_query(query)

    print("\n############ Exact-term / acronym-style queries ############")
    for query in EXACT_TERM_QUERIES:
        run_query(query)

    print("\n############ Out-of-corpus queries (expect shortcut path) ############")
    for query in OUT_OF_CORPUS_QUERIES:
        run_query(query)


if __name__ == "__main__":
    main()
