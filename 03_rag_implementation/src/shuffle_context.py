import duckdb
import random
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf

tqdm.pandas()


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

def get_id(con, references):
    values = ", ".join(
        f"('{item['document_id']}', {item['index']}, {pos})"
        for pos, item in enumerate(references)
    )

    query = f"""
        SELECT
            t.document_id,
            t.index,
            t.global_id,
            t.text
        FROM paragraph t
        INNER JOIN (VALUES {values}) AS lookup(document_id, idx, pos)
            ON t.document_id = lookup.document_id
            AND t.index = lookup.idx
        ORDER BY lookup.pos
    """

    results = con.execute(query).fetchdf().to_dict(orient='records')
    return results

def get_paragraphs(con, retrieved, k):
    if len(retrieved) == 0:
        return []

    values = ", ".join(
        f"({global_id}, {pos})"
        for pos, global_id in enumerate(retrieved[:k])
    )

    result = con.execute(f"""
        SELECT t.document_id, t.global_id, t.index, t.text
        FROM paragraph t
        INNER JOIN (VALUES {values}) AS lookup(global_id, pos)
            ON t.global_id = lookup.global_id
        ORDER BY lookup.pos
    """).df()

    return result.to_dict(orient='records')


def hybrid(results, k):
    scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            scores.setdefault(doc, 0.0)
            scores[doc] += 1.0 / (k + rank + 1)

    out = [r[0] for r in list(sorted(scores.items(), key=lambda item: item[1], reverse=True))][:k]
    return out

def shuffle(list):
    random.Random(42).shuffle(list)
    return list

def save_df(df, path):
    df[['id', 'input', 'answer', 'references', 'retrieved']].to_json(path, lines=True, orient='records')

def retrieve(cfg):
    K = cfg.K

    con = duckdb.connect(cfg.wiki_all)
    df = pd.read_json(cfg.precomputed, lines=True)
    # df = df.head(1)

    logger.info('Preprocess Precomputed Results')
    df['id'] = df['id'].astype(str)
    df['references'] = df['references'].progress_apply(lambda r: get_id(con, r))
    df['ref'] = df['references'].apply(lambda r: [element['global_id'] for element in r])
    df['hybrid'] = df.apply(lambda r: hybrid([r['dense'][:K], r['sparse'][:K]], K), axis=1)

    df['retrieved'] = df.progress_apply(lambda r: get_paragraphs(con, r['sparse'], K), axis=1)
    save_df(df, f"{cfg.out}/sparse.jsonl")
    df['retrieved'] = df.progress_apply(lambda r: get_paragraphs(con, r['dense'], K), axis=1)
    save_df(df, f"{cfg.out}/dense.jsonl")
    df['retrieved'] = df.progress_apply(lambda r: get_paragraphs(con, r['hybrid'], K), axis=1)
    save_df(df, f"{cfg.out}/hybrid.jsonl")

    df['retrieved'] = df.progress_apply(lambda r: shuffle(get_paragraphs(con, r['sparse'], K)), axis=1)
    save_df(df, f"{cfg.out}/sparse_shuffled.jsonl")

    df['retrieved'] = df.progress_apply(lambda r: shuffle(get_paragraphs(con, r['dense'], K)), axis=1)
    save_df(df, f"{cfg.out}/dense_shuffled.jsonl")

    df['retrieved'] = df.progress_apply(lambda r: shuffle(get_paragraphs(con, r['hybrid'], K)), axis=1)
    save_df(df, f"{cfg.out}/hybrid_shuffled.jsonl")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config/base.yaml"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    Path(cfg.out).mkdir(parents=True, exist_ok=True)

    retrieve(cfg)

main()



