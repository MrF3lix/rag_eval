import duckdb
import numpy as np
from retriv import set_base_path, SparseRetriever as SparseRetrieverRetriv
from pathlib import Path

from .base_retriever import BaseRetriever
from .query import Query

class SparseRetriever(BaseRetriever):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        p = Path(cfg.index.sparse.path)
        set_base_path(str(p.parent))

        self.sr = SparseRetrieverRetriv.load(p.name)

    def retriev(self, query: Query) -> Query:
        self.con = duckdb.connect(self.cfg.knowledge_base.target)

        results = self.sr.search(
            query=query.input,      
            return_docs=True,
            cutoff=self.cfg.retriever.k,
        )

        ids = np.array([r['id'] for r in results])
        result = self.con.execute("""
            SELECT *
            FROM paragraph
            WHERE global_id IN ({})
            """.format(",".join(map(str, ids)))).df()

        result = result.to_dict(orient='records')

        query.retrieved = self.results_to_paragraphs(result)

        self.con.close()

        return query
        
    def retrieve_ids(self, input):
        results = self.sr.search(
            query=input,
            cutoff=self.cfg.retriever.k,
        )

        return [r['id'] for r in results]

