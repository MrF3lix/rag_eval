import random

from .base_retriever import BaseRetriever
from .query import Query

class ProbabilisticRetrieverPrecomputed(BaseRetriever):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def retriev(self, query: Query) -> Query:

        if query.retrieved_correct_paragraph:
            rate = random.uniform(0, 1)
            keep = False
            if rate <= self.cfg.retriever.p:
                keep = True
            if not keep:
                references = [f"{r.document_id}_{r.index}" for r in query.reference]
                query.is_retrieved_adjusted = True
                query.retrieved = [r for r in query.retrieved if f"{r.document_id}_{r.index}" in references]

                # replace instead of remove


                return query
            return query
        else:
            return query