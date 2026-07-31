from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import duckdb

from .base_retriever import BaseRetriever
from .query import Query

class SimilarRetriever(BaseRetriever):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.model = SentenceTransformer(cfg.embedder.model, trust_remote_code=True)
        self.index = faiss.read_index(cfg.index.dense.path)
        self.con = duckdb.connect(cfg.knowledge_base.target)

    def retriev(self, query: Query) -> Query:
        # query.target_answer = NOT ENOUGH INFO


        multiplier = 2
        reference_paragraphs = list(map(lambda p: p.index, query.reference))

        query_embedding = self.model.encode(
            query.input,
            task=self.cfg.embedder.query_task,
            prompt_name=self.cfg.embedder.query_task,
        )


        while True:
            distances, indices = self.index.search(np.array([query_embedding]), self.cfg.retriever.k * multiplier)
            indices = indices+1 # Indices in duckdb start at 1 while indices in the faiss index start at 0

            result = self.con.execute("""
                SELECT *
                FROM paragraph
                WHERE global_id IN ({})
                """.format(",".join(map(str, indices[0])))).df()
            result['d'] = distances[0]

            result = result.loc[~result['index'].isin(reference_paragraphs)]

            if len(result) >= self.cfg.retriever.k:
                result = result.head(self.cfg.retriever.k)
                result = result.to_dict(orient='records')
                query.retrieved = self.results_to_paragraphs(result)
                break
            else:
                multiplier += 1

            if multiplier > 7:
                break

        return query
        
