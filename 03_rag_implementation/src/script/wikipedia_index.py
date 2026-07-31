import duckdb
import numpy as np
from tqdm import tqdm
from faiss import IndexHNSWFlat, write_index, METRIC_INNER_PRODUCT, omp_set_num_threads

EMBEDDER_MODEL = 'jinaai/jina-embeddings-v3'
EMBEDDER_TASK = 'text-matching'
TARGET_DB = 'data/all.duckdb'

DIM = 1024

INDEX_FILE = "data/test.index"
BATCH_SIZE = 1000

M=16
EF_CONSTRUCTION=16
NUM_THREADS=15

omp_set_num_threads(NUM_THREADS)

index = IndexHNSWFlat(DIM, M, METRIC_INNER_PRODUCT)

index.hnsw.efConstruction = EF_CONSTRUCTION


con = duckdb.connect(TARGET_DB, read_only=True)
total = con.execute("SELECT count(*) FROM paragraph WHERE has_vec = TRUE").fetchall()[0][0]
pbar = tqdm(total=total)

while True:
    batch = con.execute("""
        SELECT global_id, vec
        FROM paragraph
        WHERE has_vec = TRUE
        LIMIT ?
    """, [BATCH_SIZE]).fetchall()

    if not batch:
        print('Done')
        break

    embeddings = [x[1] for x in batch]
    embeddings = np.asarray(embeddings, dtype="float32")
    index.add(embeddings)
    pbar.update(len(embeddings))
    

write_index(index, INDEX_FILE)