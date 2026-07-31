from abc import ABC, abstractmethod

class BaseIndex(ABC):

    @abstractmethod
    def add_paragraphs(self, batch):
        pass

    @abstractmethod
    def save_index(self):
        pass