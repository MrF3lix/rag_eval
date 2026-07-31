from abc import ABC, abstractmethod
from retriever import Query


class BaseGenerator(ABC):
    """Abstract Class for the generator strategies"""

    @abstractmethod
    def generate(self, query: Query) -> Query:
        """Executes the generate method

        Parameters
        ---
        query : Query
            Query containing the input statement and the retrieved paragraphs.

        Returns
        ---
        Query
            Query with the addition of the generated answer.
        
        """
        pass