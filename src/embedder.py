"""
embedder.py  —  Phase 4: Embeddings and Vector Database
---------------------------------------------------------
Reads all 336 text chunks from text_summaries/, converts each one
into a 384-dimensional vector using a local sentence-transformer
model, and stores everything in ChromaDB.

WHY sentence-transformers (all-MiniLM-L6-v2)?
    - Runs entirely locally on your machine — no API calls, no cost
    - 384-dimensional vectors: compact but expressive enough for
      factual domain text like traffic data
    - Well-tested on domain-specific retrieval tasks
    - Embeds all 336 chunks in ~2 seconds on an M3 Mac

WHY ChromaDB?
    - Local vector database — no server setup, no cloud account
    - Stores vectors + original text + metadata together
    - Simple Python API, good fit for a project of this size
    - Persists to disk so you only embed once

WHAT GETS STORED PER CHUNK:
    - The embedding vector (384 floats)
    - The full original text (for the LLM to read in Phase 6)
    - Metadata: location_id, chunk_type, road_name, chunk_id
      (used by the retriever in Phase 5 to filter results)

OUTPUT:
    vectorstore/   ← ChromaDB persists here automatically
"""

import os
import json
import time
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import chromadb


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_NAME      = 'all-MiniLM-L6-v2'   # 384-dimensional, free, local
COLLECTION_NAME = 'traffic_guntur'      # name for the ChromaDB collection
BATCH_SIZE      = 64                    # embed this many chunks at once


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

def load_model():
    """
    Load the sentence-transformer model.
    First run downloads ~90MB to ~/.cache/huggingface/
    Subsequent runs load from cache instantly.
    """
    print(f"Loading embedding model: {MODEL_NAME}")
    print("(First run downloads ~90MB — subsequent runs use cache)\n")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}\n")
    return model


# ─────────────────────────────────────────────
# LOAD CHUNKS FROM MANIFEST
# ─────────────────────────────────────────────

def load_chunks(text_summaries_dir):
    """
    Reads the manifest to get all chunk metadata, then reads each
    .txt file to get the actual text content.

    Returns a list of dicts:
        {chunk_id, location_id, road_name, chunk_type, text}
    """
    manifest_path = os.path.join(text_summaries_dir, '_chunks_manifest.json')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}\n"
            f"Run converter.py first to generate the text chunks."
        )

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    chunks = []
    missing = []

    for entry in manifest:
        txt_path = os.path.join(text_summaries_dir, entry['file'])
        if not os.path.exists(txt_path):
            missing.append(entry['file'])
            continue
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        chunks.append({
            'chunk_id':    entry['chunk_id'],
            'location_id': entry['location_id'],
            'road_name':   entry.get('road_name', ''),
            'chunk_type':  entry['chunk_type'],
            'text':        text,
        })

    if missing:
        print(f"Warning: {len(missing)} chunk files in manifest not found on disk")

    print(f"Loaded {len(chunks)} chunks from manifest\n")
    return chunks


# ─────────────────────────────────────────────
# SETUP CHROMADB
# ─────────────────────────────────────────────

def setup_chromadb(vectorstore_dir, reset=False):
    """
    Creates (or loads) a persistent ChromaDB collection.

    reset=True  → deletes existing collection and starts fresh
    reset=False → loads existing collection if it exists
                  (safe to re-run without re-embedding everything)
    """
    os.makedirs(vectorstore_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=vectorstore_dir)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        except:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for text
    )

    existing = collection.count()
    if existing > 0:
        print(f"Collection '{COLLECTION_NAME}' already exists with {existing} vectors.")
        print("Pass reset=True to re-embed from scratch.\n")
    else:
        print(f"Created new collection '{COLLECTION_NAME}'\n")

    return client, collection


# ─────────────────────────────────────────────
# EMBED AND STORE
# ─────────────────────────────────────────────

def embed_and_store(chunks, model, collection):
    """
    Embeds all chunks in batches and upserts them into ChromaDB.

    Uses upsert (not add) so this is safe to re-run — existing
    vectors are updated rather than duplicated.
    """
    # Skip chunks already in collection
    existing_ids = set(collection.get(include=[])['ids'])
    new_chunks   = [c for c in chunks if c['chunk_id'] not in existing_ids]

    if not new_chunks:
        print("All chunks already embedded. Nothing to do.")
        print("Run with reset=True to re-embed everything.\n")
        return

    if existing_ids:
        print(f"{len(existing_ids)} chunks already in collection, "
              f"embedding {len(new_chunks)} new chunks...\n")
    else:
        print(f"Embedding {len(new_chunks)} chunks in batches of {BATCH_SIZE}...\n")

    start_time  = time.time()
    total_added = 0

    for i in tqdm(range(0, len(new_chunks), BATCH_SIZE), desc="Embedding"):
        batch = new_chunks[i : i + BATCH_SIZE]

        texts     = [c['text']       for c in batch]
        ids       = [c['chunk_id']   for c in batch]
        metadatas = [
            {
                'location_id': c['location_id'],
                'road_name':   c['road_name'],
                'chunk_type':  c['chunk_type'],
            }
            for c in batch
        ]

        # Generate embeddings — this is the core of Phase 4
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # Store in ChromaDB: vector + text + metadata together
        collection.upsert(
            ids        = ids,
            embeddings = embeddings,
            documents  = texts,       # original text stored alongside vector
            metadatas  = metadatas,
        )

        total_added += len(batch)

    elapsed = time.time() - start_time
    print(f"\nDone! Embedded {total_added} chunks in {elapsed:.1f}s")
    print(f"Total vectors in collection: {collection.count()}\n")


# ─────────────────────────────────────────────
# SANITY CHECK — test a real query
# ─────────────────────────────────────────────

def run_sanity_check(collection, model):
    """
    Runs 3 test queries against the collection to verify retrieval works.
    Prints the top result for each query so you can visually confirm
    the right chunks are being returned.
    """
    test_queries = [
        "which road has the highest truck traffic",
        "what is the peak hour at Nadikudi",
        "location with most two wheelers",
    ]

    print("=" * 60)
    print("  SANITY CHECK — Testing 3 queries")
    print("=" * 60)

    for query in test_queries:
        query_embedding = model.encode(query).tolist()

        results = collection.query(
            query_embeddings = [query_embedding],
            n_results        = 1,
            include          = ['documents', 'metadatas', 'distances'],
        )

        top_doc      = results['documents'][0][0]
        top_meta     = results['metadatas'][0][0]
        top_distance = results['distances'][0][0]
        similarity   = 1 - top_distance  # cosine distance → similarity

        print(f"\nQuery : \"{query}\"")
        print(f"Best match : {top_meta['location_id']} "
              f"[{top_meta['chunk_type']}] — {top_meta['road_name']}")
        print(f"Similarity : {similarity:.3f}")
        print(f"Preview    : {top_doc[:120].strip()}...")


# ─────────────────────────────────────────────
# PRINT COLLECTION STATS
# ─────────────────────────────────────────────

def print_stats(collection):
    """Show a breakdown of what's stored in the collection."""
    total = collection.count()

    # Count by chunk type
    chunk_types = ['overview', 'traffic', 'directional', 'peak']
    print("\n" + "=" * 60)
    print("  COLLECTION STATS")
    print("=" * 60)
    print(f"  Total vectors stored : {total}")
    print(f"  Collection name      : {COLLECTION_NAME}")
    print(f"  Similarity metric    : cosine")
    print(f"  Embedding model      : {MODEL_NAME}")
    print(f"  Embedding dimensions : 384")
    print()
    print("  Vectors by chunk type:")
    for ct in chunk_types:
        results = collection.get(where={"chunk_type": ct}, include=[])
        print(f"    {ct:<15} : {len(results['ids'])}")


# ─────────────────────────────────────────────
# RUN DIRECTLY
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Embed traffic chunks into ChromaDB")
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Wipe the existing ChromaDB collection and re-embed everything from scratch'
    )
    args = parser.parse_args()

    TEXT_DIR    = os.path.join(os.path.dirname(__file__), '..', 'text_summaries')
    VECTORSTORE = os.path.join(os.path.dirname(__file__), '..', 'vectorstore')

    # ── Step 1: Load model ──
    model = load_model()

    # ── Step 2: Load chunks ──
    chunks = load_chunks(TEXT_DIR)

    # ── Step 3: Setup ChromaDB ──
    client, collection = setup_chromadb(VECTORSTORE, reset=args.reset)

    # ── Step 4: Embed and store ──
    embed_and_store(chunks, model, collection)

    # ── Step 5: Print stats ──
    print_stats(collection)

    # ── Step 6: Sanity check ──
    run_sanity_check(collection, model)