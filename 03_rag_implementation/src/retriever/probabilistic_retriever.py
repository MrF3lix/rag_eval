import random
from .base_retriever import BaseRetriever
from .oracle_retriever import OracleRetriever
from .random_retriever import RandomRetriever
from .similar_retriever import SimilarRetriever
from .query import Query

class ProbabilisticRetriever(BaseRetriever):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.oracle = OracleRetriever(cfg)
        self.random = RandomRetriever(cfg)
        self.similar = SimilarRetriever(cfg)

    def __init__(self, cfg, oracle, random, similar):
        super().__init__()
        self.cfg = cfg

        self.oracle = oracle
        self.random = random
        self.similar = similar

    def retriev(self, query: Query) -> Query:
        rate = random.uniform(0, 1)
        if rate <= self.cfg.retriever.p:
            return self.oracle.retriev(query)
        
        if self.cfg.retriever.alternative == 'similar':
            return self.similar.retriev(query)

        return self.random.retriev(query)