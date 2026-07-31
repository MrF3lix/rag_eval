import duckdb

from .base_retriever import BaseRetriever
from .query import Query

class OracleRetriever(BaseRetriever):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.con = duckdb.connect(cfg.knowledge_base.target)

    def retriev(self, query: Query) -> Query:
        document_ids = list(map(lambda p: p.document_id, query.reference))
        reference_paragraphs = list(map(lambda p: p.index, query.reference))

        # Reference is Empty so nothing can be retrieved.
        if len(reference_paragraphs) == 0:
            return query

        result = self.con.execute(f"""
            SELECT *
            FROM paragraph
            WHERE document_id IN ({','.join(map(str, document_ids))})
            AND index IN ({','.join(map(str, reference_paragraphs))})
        """).df()

        result['d'] = 0
        result = result.to_dict(orient='records')

        query.retrieved = self.results_to_paragraphs(result)

        return query