"""
evaluate.py  —  Phase 7: RAGAS Evaluation
-------------------------------------------
Formally evaluates the RAG pipeline using three RAGAS metrics:

  Faithfulness      — does the answer stay grounded in retrieved context?
  Answer Relevancy  — does the answer actually address the question?
  Context Precision — were the retrieved chunks relevant to the question?

Usage:
    python src/evaluate.py
    python src/evaluate.py --save
"""

import math
import os
import sys
import json
import argparse
import time

from dotenv import load_dotenv
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision  # type: ignore
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from retriever import TrafficRetriever
from generator import generate_answer

BASE_DIR      = os.path.join(os.path.dirname(__file__), '..')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
VECTORSTORE   = os.path.join(BASE_DIR, 'vectorstore')
RESULTS_PATH  = os.path.join(BASE_DIR, 'docs', 'evaluation_results.json')

TEST_CASES = [
    # Superlative (5)
    {"question": "Which road in Guntur district has the highest truck traffic?",
     "reference": "P526 on the Hyderabad to Guntur Road has the highest truck traffic with 1,179 trucks per day.",
     "type": "superlative"},
    {"question": "Which location has the lowest AADT in the district?",
     "reference": "P605 on the Pakalapadu to Paladugu Via Abburu road has the lowest AADT with 63 vehicles per day.",
     "type": "superlative"},
    {"question": "Which road has the most two-wheelers per day?",
     "reference": "P528 on the Guntur to Bapatla to Chirala Road has the most two-wheelers with 7,093 per day.",
     "type": "superlative"},
    {"question": "Which location has the highest total AADT?",
     "reference": "P528 on the Guntur to Bapatla to Chirala Road has the highest AADT with 12,669 vehicles per day.",
     "type": "superlative"},
    {"question": "Which road has the highest goods vehicle traffic?",
     "reference": "P526 on the Hyderabad to Guntur Road has the highest goods vehicle traffic with 1,688 goods vehicles per day.",
     "type": "superlative"},
    # Location specific by code (3)
    {"question": "Tell me about survey location P606.",
     "reference": "P606 is on the Pulladigunta to Pericherla road near Nallapadu. AADT is 448 vehicles per day. Peak hour is 11:00-12:00.",
     "type": "location_specific_code"},
    {"question": "What is the AADT at location P605?",
     "reference": "The AADT at P605 on the Pakalapadu to Paladugu Via Abburu road is 63 vehicles per day.",
     "type": "location_specific_code"},
    {"question": "What is the peak hour at location P528?",
     "reference": "The peak hour at P528 on the Guntur to Bapatla to Chirala Road is between 18:00 and 19:00.",
     "type": "location_specific_code"},
    # Location specific by name (3)
    {"question": "What is the peak hour traffic at Nadikudi?",
     "reference": "The peak hour at location P526 near Nadikudi on the Hyderabad to Guntur Road is 18:00-19:00 with 443 vehicles.",
     "type": "location_specific_name"},
    {"question": "What is the busiest time of day on the Hyderabad to Guntur Road?",
     "reference": "The busiest time on the Hyderabad to Guntur Road at P526 is 18:00-19:00 with 443 vehicles.",
     "type": "location_specific_name"},
    {"question": "How much traffic is on the Guntur to Bapatla to Chirala Road?",
     "reference": "The Guntur to Bapatla to Chirala Road at location P528 has an AADT of 12,669 vehicles per day.",
     "type": "location_specific_name"},
    # Comparison (2)
    {"question": "Compare the traffic at P526 and P538.",
     "reference": "P526 has AADT 6,537. P538 on the Chilakaluripet to Narasaraopet Road has AADT 10,560. P538 has higher traffic.",
     "type": "comparison"},
    {"question": "Which has more traffic, P605 or P606?",
     "reference": "P606 has more traffic with 448 vehicles per day compared to P605 with 63 vehicles per day.",
     "type": "comparison"},
    # Semantic (2)
    {"question": "Which roads were surveyed near Tadikonda?",
     "reference": "P577 on the Tadikoda to Meet NH 5 via Kantheru road and P-562 on the Tadikonda to Rayapudi road.",
     "type": "semantic"},
    {"question": "Which location has the most balanced directional traffic split?",
     "reference": "A location where both direction traffic counts are nearly equal.",
     "type": "semantic"},
]


def run_pipeline(retriever, test_cases, delay=5.0):
    results = []
    total = len(test_cases)
    for i, case in enumerate(test_cases, 1):
        q = case['question']
        print(f"  [{i:2d}/{total}] {q[:60]}...")
        retrieved = retriever.retrieve(q)
        response  = generate_answer(q, retrieved)
        results.append({
            'question':  q,
            'answer':    response['answer'],
            'contexts':  [r.text for r in retrieved],
            'reference': case['reference'],
            'type':      case['type'],
            'locations': response['locations'],
            'strategy':  response['strategy'],
        })
        if i < total:
            time.sleep(delay)
    return results


def build_ragas_dataset(results):
    samples, skipped = [], 0
    for r in results:
        if r['answer'].startswith('Error:'):
            skipped += 1
            continue
        samples.append(SingleTurnSample(
            user_input         = r['question'],
            response           = r['answer'],
            retrieved_contexts = r['contexts'],
            reference          = r['reference'],
        ))
    if skipped:
        print(f"  Note: {skipped} result(s) skipped (generator returned errors — rate limit).")
        print(f"  Scoring {len(samples)} of {len(results)} results.")
    return EvaluationDataset(samples=samples), len(samples)


def extract_score(val):
    if isinstance(val, list):
        valid = [v for v in val if v is not None and not math.isnan(float(v))]
        return sum(valid) / len(valid) if valid else float('nan')
    try:
        return float(val)
    except (TypeError, ValueError):
        return float('nan')


def print_results(results, ragas_scores):
    print("\n" + "=" * 65)
    print("  EVALUATION RESULTS — Per Query")
    print("=" * 65)
    for r in results:
        ans = r['answer']
        print(f"\nQ        : {r['question']}")
        print(f"Type     : {r['type']}")
        print(f"Strategy : {', '.join(r['strategy'])}")
        print(f"Locations: {', '.join(r['locations'])}")
        print(f"Answer   : {ans[:200]}{'...' if len(ans) > 200 else ''}")

    print("\n" + "=" * 65)
    print("  RAGAS SCORES — Aggregate")
    print("=" * 65)

    metrics_order = ['faithfulness', 'answer_relevancy', 'context_precision']
    descriptions = {
        'faithfulness':      'Is the answer grounded in retrieved context?',
        'answer_relevancy':  'Does the answer address the question asked?',
        'context_precision': 'Were the retrieved chunks relevant?',
    }
    for metric in metrics_order:
        score = ragas_scores.get(metric, float('nan'))
        label = metric.replace('_', ' ').title()
        if math.isnan(score):
            print(f"\n{label}")
            print(f"  Score : N/A")
            print(f"  Means : {descriptions[metric]}")
        else:
            filled  = int(score * 20)
            bar = "\u2588" * filled + "\u2591" * (20 - filled)
            print(f"\n{label}")
            print(f"  Score : {score:.3f}  [{bar}]")
            print(f"  Means : {descriptions[metric]}")

    valid = [ragas_scores[m] for m in metrics_order
             if not math.isnan(ragas_scores.get(m, float('nan')))]
    overall = sum(valid) / len(valid) if valid else float('nan')
    print(f"\n{'─' * 65}")
    if math.isnan(overall):
        print("  Overall average : N/A")
    else:
        print(f"  Overall average : {overall:.3f}")
    print(f"{'─' * 65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7: RAGAS Evaluation")
    parser.add_argument('--save', action='store_true',
                        help='Save results to docs/evaluation_results.json')
    args = parser.parse_args()

    retriever = TrafficRetriever(PROCESSED_DIR, VECTORSTORE)

    groq_key = os.getenv('GROQ_API_KEY')
    if not groq_key:
        print("Error: GROQ_API_KEY not found in .env")
        sys.exit(1)

    groq_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key, temperature=0)
    hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    faithfulness.llm            = groq_llm
    answer_relevancy.llm        = groq_llm
    answer_relevancy.embeddings = hf_embeddings
    context_precision.llm       = groq_llm

    print("\n" + "=" * 65)
    print("  PHASE 7 — RAGAS EVALUATION")
    print(f"  {len(TEST_CASES)} test cases across 4 query types")
    print("=" * 65)
    print("\nStep 1 — Running pipeline for all test cases...")
    print("(5 second pause between calls for rate limiting)\n")

    results = run_pipeline(retriever, TEST_CASES, delay=5.0)

    print("\nStep 2 — Running RAGAS evaluation...")
    print("(RAGAS uses the LLM internally to score each response)\n")

    dataset, scored_count = build_ragas_dataset(results)
    if scored_count == 0:
        print("No scoreable results — all queries hit the daily rate limit.")
        print("The Groq free tier allows 100,000 tokens/day. Run again tomorrow.")
        sys.exit(0)

    ragas_result = evaluate(
        dataset = dataset,
        metrics = [faithfulness, answer_relevancy, context_precision],
    )

    ragas_scores = {
        'faithfulness':      extract_score(ragas_result['faithfulness']),
        'answer_relevancy':  extract_score(ragas_result['answer_relevancy']),
        'context_precision': extract_score(ragas_result['context_precision']),
    }

    print_results(results, ragas_scores)

    if args.save:
        output = {
            'ragas_scores':   ragas_scores,
            'num_test_cases': len(TEST_CASES),
            'num_scored':     scored_count,
            'model':          'llama-3.3-70b-versatile',
            'results': [{k: v for k, v in r.items() if k != 'contexts'} for r in results],
        }
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {RESULTS_PATH}")