from abc import ABC, abstractmethod


class AbstractDocumentLoader(ABC):
    @abstractmethod
    def load(self, doc_name: str) -> str:
        pass