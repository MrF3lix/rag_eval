import json
import logging
import argparse
import duckdb
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf

from retriever import DenseRetriever, SparseRetriever, OracleRetriever, HybridRetriever, Query

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
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)

    run_test_queries(cfg)

def run_test_queries(cfg):
    der = DenseRetriever(cfg)
    spr = SparseRetriever(cfg)

    num_queries = sum(1 for _ in open(cfg.documents.target))
    with open(cfg.documents.target) as f, open(cfg.output, 'w') as out:
        for line in tqdm(f, total=num_queries):
            query = json.loads(line)

            query_spr = spr.retrieve_ids(query['input'])
            query_der = der.retrieve_ids(query['input'])

            # TODO: Context Transplant

            json.dump({
                'id': query['id'],
                'input': query['input'],
                'answer': query['answer'],
                'references': query['references'],
                'sparse': query_spr,
                'dense': query_der
            }, out)
            out.write('\n')

main()