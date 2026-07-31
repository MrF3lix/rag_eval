import duckdb

from .base_retriever import BaseRetriever
from .query import Paragraph, Query

class RandomRetriever(BaseRetriever):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.con = duckdb.connect(cfg.knowledge_base.target)

    def retriev(self, query: Query) -> Query:
        reference_documents = list(map(lambda p: p.document_id, query.reference))

        # Reference is Empty so no document should be removed from the query (No document with ID -1 exists)
        if len(reference_documents) == 0:
            reference_documents.append(-1)

        result = self.con.execute(f"""
            SELECT *
            FROM paragraph
            WHERE document_id NOT IN ({','.join(['?']*len(reference_documents))})
            USING SAMPLE 5;
        """, reference_documents).df()

        result = result.to_dict(orient='records')
        query.retrieved = self.results_to_paragraphs(result)

        return query
