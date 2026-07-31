from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import duckdb

from .base_retriever import BaseRetriever
from .query import Query

class DenseRetriever(BaseRetriever):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.model = SentenceTransformer(cfg.embedder.model, trust_remote_code=True)
        self.index = faiss.read_index(cfg.index.dense.path)

    def retriev(self, query: Query) -> Query:
        self.con = duckdb.connect(self.cfg.knowledge_base.target)
        query_embedding = self.model.encode(
            query.input,
            task=self.cfg.embedder.query_task,
            prompt_name=self.cfg.embedder.query_task,
        )

        distances, indices = self.index.search(np.array([query_embedding]), self.cfg.retriever.k)
        indices = indices+1 # Indices in duckdb start at 1 while indices in the faiss index start at 0

        result = self.con.execute("""
            SELECT *
            FROM paragraph
            WHERE global_id IN ({})
            """.format(",".join(map(str, indices[0])))).df()

        result['d'] = distances[0]
        # TODO: Find out why the distance is always the same? is this related to the index type?

        result = result.to_dict(orient='records')

        query.retrieved = self.results_to_paragraphs(result)

        self.con.close()

        return query
        
        
    def retrieve_ids(self, input):
        query_embedding = self.model.encode(
            input,
            task=self.cfg.embedder.query_task,
            prompt_name=self.cfg.embedder.query_task,
        )

        distances, indices = self.index.search(np.array([query_embedding]), self.cfg.retriever.k)
        indices = indices+1 # Indices in duckdb start at 1 while indices in the faiss index start at 0

        return indices.tolist()[0]
