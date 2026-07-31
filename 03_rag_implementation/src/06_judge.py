import json
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime
from itertools import islice
from concurrent.futures import ThreadPoolExecutor

from retriever import Query
from judge import LLMJudge

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config/base.yaml"
    )
    parser.add_argument(
        "--output", type=str, help="Path to output folder", default="results"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    base = OmegaConf.load(cfg.base)
    cfg = OmegaConf.merge(base, cfg)

    now = datetime.today().strftime('%Y-%m-%d_%H-%M')
    report_path = f"{args.output}/{now}_{cfg.name}"
    Path(report_path).mkdir(parents=True, exist_ok=True)
    logger.debug(f'Started {now}')

    with open(f'{report_path}/config.yaml', 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    results = run_test_queries(cfg)
    with open(f'{report_path}/results.jsonl', 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

def run_test_queries(cfg):
    judge = LLMJudge(cfg)

    def process_query(line):
        query = Query.model_validate_json(line)

        query = judge.evaluate(query)

        return query.compute_result()

    num_queries = sum(1 for _ in open(cfg.documents.target))
    results = []
    with open(cfg.documents.target) as f, ThreadPoolExecutor(max_workers=cfg.judge.max_threads) as executor:
        # futures = executor.map(process_query, islice(f, 1))
        futures = executor.map(process_query, f)
        for result in tqdm(futures, total=num_queries):
            results.append(result)

    return results

main()