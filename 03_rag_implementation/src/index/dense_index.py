from sentence_transformers import SentenceTransformer
from faiss import IndexHNSWFlat, write_index, METRIC_INNER_PRODUCT

from .base_index import BaseIndex

class DenseIndex(BaseIndex):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.embedder.model, trust_remote_code=True)
        self.index = IndexHNSWFlat(cfg.index.dim, 32, METRIC_INNER_PRODUCT)

    def add_paragraphs(self, batch):
        texts = [r[1] for r in batch]

        embeddings = self.model.encode(
            texts,
            task=self.cfg.embedder.task,
            prompt_name=self.cfg.embedder.task,
        )

        self.index.add(embeddings.astype("float32"))

    def save_index(self):
        write_index(self.index, self.cfg.index.dense.path)