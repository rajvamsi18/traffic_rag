"""
diagnose.py  —  Check whether ranking sentences are in chunks and ChromaDB
Run: python src/diagnose.py
"""
import os, json
import chromadb
from sentence_transformers import SentenceTransformer

BASE          = os.path.join(os.path.dirname(__file__), '..')
TEXT_DIR      = os.path.join(BASE, 'text_summaries')
PROCESSED_DIR = os.path.join(BASE, 'data', 'processed')
VECTORSTORE   = os.path.join(BASE, 'vectorstore')

print("=" * 60)
print("STEP 1 — Verify chunk files are clean (no ranking sentences)")
print("=" * 60)

p526_traffic = os.path.join(TEXT_DIR, 'P526_traffic.txt')
if os.path.exists(p526_traffic):
    with open(p526_traffic) as f:
        content = f.read()
    if 'ranks' not in content.lower():
        print("✅ P526_traffic.txt is clean — no ranking sentence present")
        print("   Last 3 lines:")
        for line in content.strip().split('\n')[-3:]:
            if line.strip():
                print(f"   {line.strip()}")
    else:
        print("⚠️  P526_traffic.txt still contains a ranking sentence")
        print("   converter.py may not have been saved correctly")
        for line in content.split('\n'):
            if 'ranks' in line.lower():
                print(f"   → {line.strip()}")
else:
    print(f"❌ File not found: {p526_traffic}")

print()
print("=" * 60)
print("STEP 2 — Check what ChromaDB currently has for P526")
print("=" * 60)

client     = chromadb.PersistentClient(path=VECTORSTORE)
collection = client.get_collection('traffic_guntur')

result = collection.get(
    ids     = ['P526_traffic'],
    include = ['documents', 'metadatas']
)

if result['ids']:
    doc = result['documents'][0]
    if 'ranks' in doc.lower():
        print("⚠️  ChromaDB P526_traffic still has OLD vector with ranking sentence")
        print("   Step 3 will reset and re-embed with clean chunks")
    else:
        print("✅ ChromaDB P526_traffic already has clean vector (no ranking sentence)")
        print("   Step 3 will still reset to ensure all 336 vectors are consistent")
else:
    print("❌ P526_traffic not found in ChromaDB at all")

print()
print("=" * 60)
print("STEP 3 — Force reset ChromaDB and re-embed with clean chunks")
print("=" * 60)
print("Running reset + re-embed...")

model = SentenceTransformer('all-MiniLM-L6-v2')

manifest_path = os.path.join(TEXT_DIR, '_chunks_manifest.json')
with open(manifest_path) as f:
    manifest = json.load(f)

chunks = []
for entry in manifest:
    fpath = os.path.join(TEXT_DIR, entry['file'])
    with open(fpath) as f:
        text = f.read().strip()
    chunks.append({
        'chunk_id':    entry['chunk_id'],
        'location_id': entry['location_id'],
        'road_name':   entry.get('road_name', ''),
        'chunk_type':  entry['chunk_type'],
        'text':        text,
    })

print(f"Loaded {len(chunks)} chunks from manifest")

# Delete and recreate collection
try:
    client.delete_collection('traffic_guntur')
    print("Deleted existing collection")
except:
    pass

collection = client.get_or_create_collection(
    name     = 'traffic_guntur',
    metadata = {"hnsw:space": "cosine"}
)

# Re-embed in batches
from tqdm import tqdm
BATCH = 64
for i in tqdm(range(0, len(chunks), BATCH), desc="Re-embedding"):
    batch = chunks[i:i+BATCH]
    collection.upsert(
        ids        = [c['chunk_id']   for c in batch],
        embeddings = model.encode([c['text'] for c in batch]).tolist(),
        documents  = [c['text']       for c in batch],
        metadatas  = [{'location_id': c['location_id'],
                       'road_name':   c['road_name'],
                       'chunk_type':  c['chunk_type']} for c in batch],
    )

print(f"\n✅ Done. {collection.count()} vectors stored.")
print("Now run: python src/retriever.py")