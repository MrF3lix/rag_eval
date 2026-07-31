from abc import ABC, abstractmethod
from retriever import Query


class BaseJudge(ABC):
    """Abstract Class for the judge strategies"""

    @abstractmethod
    def evaluate(self, query: Query) -> Query:
        """Executes the generate method

        Parameters
        ---
        query : Query
            Query containing the reference and retrieved documents as well as the reference answer and the generated answer.

        Returns
        ---
        Query
            Query with the addition of the evaluation results.
        
        """
        pass
    
    def retrieved_correct_document(self, query):
        reference_document_ids = [r.document_id for r in query.references]
        retrieved_document_ids = [r.document_id for r in query.retrieved]

        return bool(set(reference_document_ids) & set(retrieved_document_ids))

    def retrieved_correct_paragraph(self, query):
        reference_paragraphs_ids = [f"{r.document_id}.{r.index}" for r in query.references]
        retrieved_paragraphs_ids = [f"{r.document_id}.{r.index}" for r in query.retrieved]

        return bool(set(reference_paragraphs_ids) & set(retrieved_paragraphs_ids))