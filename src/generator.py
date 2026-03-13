"""
generator.py  —  Phase 6: Answer Generation
---------------------------------------------
Takes retrieved chunks from the Phase 5 retriever and sends them
to the Groq API along with the user's question. Returns a grounded
natural language answer based strictly on the retrieved context.

WHY PROMPT DESIGN MATTERS MORE THAN CODE HERE:
    The biggest failure mode in RAG is hallucination — the LLM answering
    confidently from its own training knowledge rather than the retrieved
    context. The prompt explicitly instructs the model to:
      - Answer only from the provided context chunks
      - Acknowledge clearly when the context is insufficient
      - Never mix in general world knowledge about Indian roads or traffic
    This is enforced through the system prompt, not through post-processing.

WHY GROQ:
    Originally built for Gemini 2.0 Flash, but Gemini's free tier daily
    quota proved too restrictive for development and testing (hitting limits
    after ~10 calls). Groq offers llama-3.1-8b-instant on a generous free
    tier with no daily cap and 30 requests/minute. The API follows the
    OpenAI-compatible format so the change was minimal — just a different
    URL, key, and model name. Prompt structure is identical.

GROQ MODEL:
    Uses llama-3.3-70b-versatile — strong reasoning, factual, free tier.
    API key is read from the .env file (GROQ_API_KEY).
"""

import os
import time
import json
import requests
from typing import List
from dotenv import load_dotenv

from retriever import RetrievalResult

load_dotenv()


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL   = 'llama-3.3-70b-versatile'
GROQ_URL     = 'https://api.groq.com/openai/v1/chat/completions'
MAX_TOKENS   = 1024


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
#
# This is the most carefully designed part of Phase 6.
# Every instruction exists to prevent a specific failure mode.

SYSTEM_PROMPT = """You are a traffic data analyst assistant for Guntur District, Andhra Pradesh.
You answer questions about classified traffic volume count surveys conducted on state highways.

STRICT RULES — follow these without exception:

1. Answer ONLY using the context chunks provided. Do not use any outside knowledge about roads,
   traffic patterns, or Andhra Pradesh geography.

2. If the context does not contain enough information to answer the question, say clearly:
   "The available survey data does not contain enough information to answer this question."
   Do not guess or infer beyond what the context states.

3. When quoting specific numbers (AADT, vehicle counts, percentages), always mention the
   location ID and road name so the answer is traceable.

4. For comparisons, only compare locations whose data is present in the context.
   Do not rank or compare locations not mentioned in the provided chunks.

5. If a retrieval note is present in the context (lines starting with [Retrieval note:]),
   use that information to explain why a particular location was selected.

6. Keep answers concise and factual. Use plain English — avoid jargon unless explaining
   a term that appears in the context (like AADT or PCU).

7. If multiple locations are in the context, structure your answer clearly by location.

The data covers 84 state highway survey locations in Guntur District, surveyed in April 2017.
All AADT figures have been adjusted using Seasonal Correction Factors to represent annual averages.
"""


# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────

def build_prompt(question: str, results: List[RetrievalResult]) -> str:
    """
    Assembles the full prompt from retrieved chunks and the user question.

    The prompt structure is:
      - Context section (retrieved chunks, labelled by location and chunk type)
      - The user's question
      - An explicit instruction to answer from context only

    The system prompt is passed separately as the 'system' role in the
    Groq/OpenAI message format, keeping it cleanly separated from context.
    Labelling each chunk with its location and type helps the model attribute
    its answer correctly rather than treating all context as one blob.
    """
    if not results:
        return (
            f"No relevant survey data was found for this query.\n\n"
            f"Question: {question}\n\n"
            f"Answer: The available survey data does not contain information "
            f"relevant to this question."
        )

    # Build context section — each chunk labelled clearly
    context_parts = []
    seen = set()

    for r in results:
        label = f"[{r.location_id} — {r.chunk_type.upper()} — {r.road_name}]"
        if label not in seen:
            seen.add(label)
            context_parts.append(f"{label}\n{r.text}")

    context_block = "\n\n---\n\n".join(context_parts)

    # Note the retrieval strategy used
    strategies = list(dict.fromkeys(r.strategy for r in results))
    strategy_note = f"[Retrieval strategy: {', '.join(strategies)}]"

    prompt = (
        f"{'='*60}\n"
        f"RETRIEVED CONTEXT\n"
        f"{strategy_note}\n"
        f"{'='*60}\n\n"
        f"{context_block}\n\n"
        f"{'='*60}\n"
        f"QUESTION: {question}\n"
        f"{'='*60}\n\n"
        f"Answer the question using only the context provided above. "
        f"If the context does not contain the answer, say so clearly."
    )

    return prompt


# ─────────────────────────────────────────────
# GROQ API CALL
# ─────────────────────────────────────────────

def call_llm(prompt: str, retries: int = 3) -> str:
    """
    Sends the prompt to the Groq API and returns the response text.
    Uses the OpenAI-compatible chat completions format.
    Retries up to 3 times on rate limit errors with a 65 second wait.
    """
    if not GROQ_API_KEY:
        return (
            "Error: GROQ_API_KEY not found in .env file.\n"
            "Add your key: GROQ_API_KEY=your_key_here"
        )

    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens":  MAX_TOKENS,
        "temperature": 0.1,
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                GROQ_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                try:
                    return data['choices'][0]['message']['content'].strip()
                except (KeyError, IndexError):
                    return f"Error: Unexpected response structure:\n{json.dumps(data, indent=2)}"

            elif response.status_code == 429:
                if attempt < retries:
                    wait = 65
                    print(f"  Rate limit hit — waiting {wait}s before retry "
                          f"(attempt {attempt}/{retries})...")
                    time.sleep(wait)
                    continue
                return "Error: Rate limit reached after all retries. Wait a minute and try again."

            elif response.status_code == 401:
                return "Error: Invalid API key. Check your GROQ_API_KEY in .env"

            else:
                return (
                    f"Error: Groq API returned status {response.status_code}\n"
                    f"{response.text[:300]}"
                )

        except requests.exceptions.Timeout:
            return "Error: Request timed out. Check your internet connection."

        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Groq API. Check your internet connection."

    return "Error: All retry attempts exhausted."


# ─────────────────────────────────────────────
# MAIN GENERATOR FUNCTION
# ─────────────────────────────────────────────

def generate_answer(question: str, results: List[RetrievalResult]) -> dict:
    """
    Full pipeline: build prompt → call LLM → return structured response.

    Returns a dict with:
        answer       → the LLM's response text
        question     → the original question
        strategy     → retrieval strategy used
        locations    → list of location IDs used as context
        chunk_types  → list of chunk types used
        prompt       → the full prompt (useful for debugging)
    """
    prompt = build_prompt(question, results)
    answer = call_llm(prompt)

    return {
        'answer':      answer,
        'question':    question,
        'strategy':    list(dict.fromkeys(r.strategy for r in results)),
        'locations':   list(dict.fromkeys(r.location_id for r in results)),
        'chunk_types': list(dict.fromkeys(r.chunk_type for r in results)),
        'num_chunks':  len(results),
        'prompt':      prompt,
    }