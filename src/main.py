"""
main.py  —  Interactive RAG Pipeline
--------------------------------------
Ties all phases together into a working question-answering system.

Pipeline:
    User question
        → Phase 5: TrafficRetriever  (classify + retrieve)
        → Phase 6: generate_answer   (prompt + Gemini API)
        → Answer printed to terminal

Usage:
    python src/main.py                    # interactive mode
    python src/main.py --test             # runs built-in test queries
    python src/main.py --debug            # shows retrieved chunks + prompt
"""

import os
import sys
import time
import argparse

from retriever  import TrafficRetriever
from generator  import generate_answer


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

BASE_DIR      = os.path.join(os.path.dirname(__file__), '..')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
VECTORSTORE   = os.path.join(BASE_DIR, 'vectorstore')


# ─────────────────────────────────────────────
# TEST QUERIES
# covers all four retrieval strategies
# ─────────────────────────────────────────────

TEST_QUERIES = [
    # Superlative
    "Which road in Guntur district has the highest truck traffic?",
    "Which location has the lowest AADT?",
    "Which road has the most two-wheelers?",

    # Location specific — by code
    "Tell me about survey location P606.",

    # Location specific — by name
    "What is the peak hour traffic at Nadikudi?",

    # Comparison
    "Compare the traffic at P526 and P538.",

    # General semantic
    "Which roads were surveyed near Tadikonda?",
    "What is the busiest time of day on the Hyderabad to Guntur Road?",
]


# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────

def print_separator(char='─', width=65):
    print(char * width)


def print_result(response: dict, debug: bool = False):
    """Prints the answer and optional debug info cleanly."""
    print_separator()
    print(f"Q: {response['question']}")
    print_separator()
    print(f"Strategy  : {', '.join(response['strategy'])}")
    print(f"Locations : {', '.join(response['locations'])}")
    print(f"Chunks    : {response['num_chunks']} ({', '.join(response['chunk_types'])})")
    print_separator()
    print(f"\n{response['answer']}\n")

    if debug:
        print_separator('·')
        print("DEBUG — Full prompt sent to Gemini:")
        print_separator('·')
        print(response['prompt'])
        print_separator('·')


# ─────────────────────────────────────────────
# TEST MODE
# ─────────────────────────────────────────────

def run_tests(retriever: TrafficRetriever, debug: bool = False):
    """Runs all test queries and prints results."""
    print("\n" + "=" * 65)
    print("  TRAFFIC RAG — TEST MODE")
    print(f"  Running {len(TEST_QUERIES)} test queries")
    print("=" * 65 + "\n")

    for i, question in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}]")
        results  = retriever.retrieve(question)
        response = generate_answer(question, results)
        print_result(response, debug=debug)
        print()
        if i < len(TEST_QUERIES):
            time.sleep(5)  # 5 second pause between queries — stays within free tier limits


# ─────────────────────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────────────────────

def run_interactive(retriever: TrafficRetriever, debug: bool = False):
    """
    Interactive question-answering loop.
    Type a question and get an answer grounded in the survey data.
    Type 'quit' or 'exit' to stop. Type 'test' to run test queries.
    """
    print("\n" + "=" * 65)
    print("  TRAFFIC RAG — Guntur District State Highway Survey Data")
    print("  84 locations · April 2017 · Andhra Pradesh")
    print("=" * 65)
    print("\nAsk any question about traffic survey data in Guntur District.")
    print("Type 'test' to run built-in test queries.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not question:
            continue

        if question.lower() in ('quit', 'exit', 'q'):
            print("Exiting.")
            break

        if question.lower() == 'test':
            run_tests(retriever, debug=debug)
            continue

        results  = retriever.retrieve(question)
        response = generate_answer(question, results)
        print_result(response, debug=debug)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Traffic RAG — Guntur District survey data Q&A'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run built-in test queries instead of interactive mode'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Show retrieved chunks and full prompt alongside each answer'
    )
    args = parser.parse_args()

    # Initialise retriever (loads JSON + embedding model + ChromaDB)
    retriever = TrafficRetriever(PROCESSED_DIR, VECTORSTORE)

    if args.test:
        run_tests(retriever, debug=args.debug)
    else:
        run_interactive(retriever, debug=args.debug)