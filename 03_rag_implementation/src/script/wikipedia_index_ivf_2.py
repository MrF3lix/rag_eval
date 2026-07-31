import duckdb
import numpy as np
from tqdm import tqdm
import faiss
import pyarrow.parquet as pq
import pyarrow as pa 

INDEX_FILE = "data/test_ivf_trained.index"
INDEX_OUT_FILE = "data/wiki_ivf.index"
TARGET_DB = 'data/all.duckdb'
TEMP_TRAINING_EMBEDDINGS = 'data/training_embeddings.npz'
DIM = 1024
BATCH_SIZE = 250_000
# NLIST = 65_536
NLIST = 16_384
NPROBE = 16
NUM_THREADS = 20


# 1. Generate Embeddings from TARGET_DB
# 2. Write Embeddings in Order to a Parquet File
# 3. Use 2.5M to train an Index
# 4. Write Trained Index to Disk
# 5. Add all rows to the Index and write filled index to disk.


faiss.omp_set_num_threads(NUM_THREADS)

con = duckdb.connect(TARGET_DB, read_only=True)

total = con.execute("""
    SELECT count(*) FROM paragraph WHERE has_vec = TRUE
""").fetchone()[0]


# train_embeddings = np.load(TEMP_TRAINING_EMBEDDINGS)
# train_embeddings = train_embeddings['arr_0']

# quantizer = faiss.IndexFlatIP(DIM)
# index = faiss.IndexIVFFlat(
#     quantizer,
#     DIM,
#     NLIST,
#     faiss.METRIC_INNER_PRODUCT
# )

# print('Start Training')

# index.train(train_embeddings)

# print('Saved Training Embeddings')
# faiss.write_index(index, INDEX_FILE)

pf = pq.ParquetFile("embeddings.parquet")
index = faiss.read_index(INDEX_FILE)


pbar = tqdm(total=total)
for batch in pf.iter_batches(batch_size=BATCH_SIZE):
    vectors = np.vstack(
        batch.column(0).to_numpy(zero_copy_only=False)
    ).astype("float32", copy=False)
    index.add(vectors)
    pbar.update(BATCH_SIZE)


index.nprobe = NPROBE
faiss.write_index(index, INDEX_OUT_FILE)