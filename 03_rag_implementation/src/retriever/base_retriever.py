from abc import ABC, abstractmethod
from .query import Query, Paragraph
 
class BaseRetriever(ABC):
    """Abstract Class for the retriever strategies"""

    @abstractmethod
    def retriev(self, query: Query) -> Query:
        """Executes the quantification method

        Parameters
        ---
        query : Query
            Uses the input of the query to search the index

        Returns
        ---
        Query
            Returns the query with a set list of retrieved paragraphs
        
        """
        pass

    def results_to_paragraphs(self, result: list[dict]) -> list[Paragraph]:
        return list(map(lambda r: Paragraph(
            document_id=r['document_id'],
            global_id=r['global_id'],
            index=r['index'],
            text=r['text'],
        ), result))