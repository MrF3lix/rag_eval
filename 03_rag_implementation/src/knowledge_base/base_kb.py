from abc import ABC, abstractmethod

class KnowledgeBase(ABC):
    @abstractmethod
    def init_database(self):
        pass

    def init_index(self):
        pass
