import duckdb
import numpy as np
from tqdm import tqdm
import faiss

INDEX_FILE = "data/test_ivf.index"
TARGET_DB = 'data/all.duckdb'
TEMP_TRAINING_EMBEDDINGS = 'data/training_embeddings.npx'
DIM = 1024
QUERY_BATCH_SIZE = 100_000
BATCH_SIZE = 1000
NLIST = 65536
NPROBE = 16
NUM_THREADS = 15
TRAINING_SIZE = 2_555_904

faiss.omp_set_num_threads(NUM_THREADS)

con = duckdb.connect(TARGET_DB, read_only=True)

total = con.execute("""
    SELECT count(*) FROM paragraph WHERE has_vec = TRUE
""").fetchone()[0]

quantizer = faiss.IndexFlatIP(DIM)
index = faiss.IndexIVFFlat(
    quantizer,
    DIM,
    NLIST,
    faiss.METRIC_INNER_PRODUCT
)

print('Prepare Query')
cursor = con.execute("""
    SELECT vec FROM paragraph
    WHERE has_vec = TRUE
""")


pbar = tqdm(total=TRAINING_SIZE)
print('Start Fetching')
train_embeddings = []
processed = 0 
while True:
    rows = cursor.fetchmany(QUERY_BATCH_SIZE)
    if not rows:
        break

    embeddings = [x[0] for x in rows]
    embeddings = np.asarray(embeddings, dtype="float32")

    train_embeddings.extend(embeddings)
    pbar.update(len(embeddings))

    processed += len(embeddings)
    if processed > TRAINING_SIZE:
        break

print('Compress and Save')
train_embeddings = np.asarray(train_embeddings, dtype="float32")
np.savez_compressed(TEMP_TRAINING_EMBEDDINGS, train_embeddings)

print('Saved Training Embeddings')

index.train(train_embeddings)

pbar = tqdm(total=total)
offset = 0

while True:
    batch = con.execute("""
        SELECT vec FROM paragraph
        WHERE has_vec = TRUE
        LIMIT ? OFFSET ?
    """, [BATCH_SIZE, offset]).fetchall()

    if not batch:
        break

    embeddings = np.asarray([x[0] for x in batch], dtype="float32")
    index.add(embeddings)

    offset += len(embeddings)
    pbar.update(len(embeddings))

index.nprobe = NPROBE
faiss.write_index(index, INDEX_FILE)
