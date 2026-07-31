import json
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime

from retriever import DenseRetriever, SparseRetriever, OracleRetriever, RandomRetriever, SimilarRetriever, HybridRetriever, ProbabilisticRetriever, Query
from generator import Generator
from judge import DefaultJudge, LLMJudge

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config/base.yaml"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    base = OmegaConf.load(cfg.base)

    cfg = OmegaConf.merge(base, cfg)

    now = datetime.today().strftime('%Y-%m-%d_%H-%M')
    report_path = f"results/{now}_{cfg.name}"
    Path(report_path).mkdir(parents=True, exist_ok=True)
    logger.debug(f'Started {now}')

    with open(f"{report_path}/config.yaml", 'w') as f:
        OmegaConf.save(cfg, f)

    results = run_test_queries(cfg)

    df = pd.DataFrame(results)
    logger.debug(f'Number of Test Queries:      {len(df)}')
    logger.debug(f'Correct Documents:           {df['correct_document'].sum() / len(df)}')
    logger.debug(f'Correct Paragraph:           {df['correct_paragraph'].sum() / len(df)}')
    logger.debug(f'Correct Answer:              {df['correct_answer'].sum() / len(df)}')

    with open(f'{report_path}/results.json', 'w') as f:
        json.dump(results, f)

def run_test_queries(cfg):
    retriever = load_retriever(cfg)
    generator = Generator(cfg)
    judge = load_judge(cfg)

    num_queries = sum(1 for _ in open(cfg.documents.target))
    results = []
    with open(cfg.documents.target) as f:
        for line in tqdm(f, total=num_queries):
            query = Query.model_validate_json(line)
            query.is_query_correct = cfg.knowledge_base.unseen == False

            query = retriever.retriev(query)
            query = generator.generate(query)
            query = judge.evaluate(query)

            results.append(query.compute_result())

    return results

def load_judge(cfg):
    if 'judge' in cfg and 'type' in cfg.judge and cfg.judge.type == 'llm':
        return LLMJudge(cfg)

    return DefaultJudge(cfg)

def load_retriever(cfg):
    if cfg.retriever.strategy == 'random':
        return RandomRetriever(cfg)
    elif cfg.retriever.strategy == 'similar':
        return SimilarRetriever(cfg)
    elif cfg.retriever.strategy == 'oracle':
        return OracleRetriever(cfg)
    elif cfg.retriever.strategy == 'sparse':
        return SparseRetriever(cfg)
    elif cfg.retriever.strategy == 'hybrid':
        return HybridRetriever(cfg)
    elif cfg.retriever.strategy == 'probabilistic':
        return ProbabilisticRetriever(cfg)

    return DenseRetriever(cfg)

main()