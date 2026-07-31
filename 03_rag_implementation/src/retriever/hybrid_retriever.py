import copy
from .base_retriever import BaseRetriever
from .sparse_retriever import SparseRetriever
from .dense_retriever import DenseRetriever
from .query import Query, Paragraph

class HybridRetriever(BaseRetriever):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.sparse = SparseRetriever(cfg)
        self.dense = DenseRetriever(cfg)

    def __init__(self, cfg, dense, sparse):
        super().__init__()
        self.cfg = cfg

        self.sparse = sparse
        self.dense = dense

    def retriev(self, query: Query) -> Query:

        query_sr = self.sparse.retriev(copy.deepcopy(query))
        query_dr = self.dense.retriev(copy.deepcopy(query))

        rrf = self.reciprocal_rank_fusion([query_sr.retrieved, query_dr.retrieved])
        rrf_keys = list(rrf.keys())[:self.cfg.retriever.k]

        retrieved = []
        seen = set()

        for p in query_sr.retrieved + query_dr.retrieved:
            if p.global_id in rrf_keys and p.global_id not in seen:
                retrieved.append(p)
                seen.add(p.global_id)

        query.retrieved = retrieved
        return query
    

    def reciprocal_rank_fusion(self, results: list[list[Paragraph]], k=60):
        scores = {}

        for docs in results:
            for rank, doc in enumerate(docs):
                scores.setdefault(doc.global_id, 0.0)
                scores[doc.global_id] += 1.0 / (k + rank + 1)

        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
