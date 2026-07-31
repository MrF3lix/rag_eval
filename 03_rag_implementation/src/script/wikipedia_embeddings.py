import json
import duckdb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

EMBEDDER_MODEL = 'jinaai/jina-embeddings-v3'
EMBEDDER_TASK = 'text-matching'
TARGET_DB = 'data/all.duckdb'

BATCH_SIZE = 250
OUTPUT_FILE = "data/embeddings.jsonl"

def append_embeddings(results):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for pid, emb in results:
            record = {
                "global_id": pid,
                "embedding": emb.tolist()
            }
            f.write(json.dumps(record) + "\n")
        f.flush()

def write_batch(con, results):
    con.execute("""
        CREATE TEMP TABLE batch_embeddings (
            id INTEGER,
            embedding FLOAT[]
        )
    """)

    con.executemany(
        "INSERT INTO batch_embeddings VALUES (?, ?)",
        results,
    )

    con.execute("""
        UPDATE paragraph
        SET vec = batch_embeddings.embedding, has_vec = TRUE
        FROM batch_embeddings
        WHERE paragraph.global_id = batch_embeddings.id
    """)

    con.execute("DROP TABLE batch_embeddings")

con = duckdb.connect(TARGET_DB)
model = SentenceTransformer(EMBEDDER_MODEL, trust_remote_code=True)

total = con.execute("SELECT count(*) FROM paragraph WHERE has_vec = FALSE").fetchall()[0][0]
pbar = tqdm(total=total)

while True:
    batch = con.execute("""
        SELECT global_id, text
        FROM paragraph
        WHERE has_vec = FALSE
        LIMIT ?
    """, [BATCH_SIZE]).fetchall()

    if not batch:
        print('Done')
        break

    ids = [r[0] for r in batch]
    texts = [r[1] for r in batch]

    embeddings = model.encode(
        texts,
        task=EMBEDDER_TASK,
        prompt_name=EMBEDDER_TASK,
    )

    write_batch(con, zip(ids, embeddings))
    pbar.update(BATCH_SIZE)