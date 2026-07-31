import random
import duckdb
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
    results = list({d["global_id"]: d for d in results}.values())
    return results

def get_global_ids(con, references):
    elements = get_id(con, references)
    return [element['global_id'] for element in elements]

def get_paragraph_with_global_id(con, references):
    elements = get_id(con, references)
    # return [element['global_id'] for element in elements]

    return elements

def hybrid(results, k):
    scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            scores.setdefault(doc, 0.0)
            scores[doc] += 1.0 / (k + rank + 1)

    out = [r[0] for r in list(sorted(scores.items(), key=lambda item: item[1], reverse=True))][:k]
    return out

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

def save_df(df, path):
    df[['id', 'input', 'answer', 'references', 'retrieved']].to_json(path, lines=True, orient='records')

def oracle_intersection(row, retriever='sparse', cutoff=5):
    ref = set(row['ref'])
    ret = set(row[retriever][:cutoff])

    return list(ref & ret)

def get_random_documents(con, limit, ommit=[]):
    result = con.execute(f"""
        SELECT global_id
        FROM paragraph
        WHERE global_id NOT IN ({','.join(['?']*len(ommit))})
        USING SAMPLE {limit};
    """, ommit).fetchall()

    global_ids = [row[0] for row in result]

    return global_ids

def probabilistic_retriever(con, row, p, cutoff, fill='random'):
    ref = list(set(row['ref']))
    context = []
    rest = []

    rate = random.uniform(0, 1)
    if rate <= (p/100):
        context = ref

    if fill == 'random':
        rest = get_random_documents(con, cutoff-len(context), ref)
    elif fill == 'dense':
        dense = row['dense'][:(cutoff*2)]
        dense = [item for item in dense if item not in ref]
        rest = dense[:cutoff-len(context)]
    elif fill == 'sparse':
        sparse = row['sparse'][:(cutoff*2)]
        sparse = [item for item in sparse if item not in ref]
        rest = sparse[:cutoff-len(context)]

    return context + rest


def retrieve(cfg):
    K = cfg.K

    con = duckdb.connect(cfg.wiki_all)
    df = pd.read_json(cfg.precomputed, lines=True)
    # df = df.sample(1000, random_state=42)

    logger.info('Preprocess Precomputed Results')
    df['id'] = df['id'].astype(str)
    df['references'] = df['references'].progress_apply(lambda r: get_paragraph_with_global_id(con, r))
    df['ref'] = df['references'].apply(lambda r: [element['global_id'] for element in r])

    for i in range(0, 110, 10):
        df['p_dense'] = df.progress_apply(lambda r: probabilistic_retriever(con, r, i, K, fill='dense'), axis=1)
        df['retrieved'] = df.progress_apply(lambda r: get_paragraphs(con, r['p_dense'], K), axis=1)
        save_df(df, f"{cfg.out}/probabilistic_{i}_dense.jsonl")

    for i in range(0, 110, 10):
        df['p_sparse'] = df.progress_apply(lambda r: probabilistic_retriever(con, r, i, K, fill='sparse'), axis=1)
        df['retrieved'] = df.progress_apply(lambda r: get_paragraphs(con, r['p_sparse'], K), axis=1)
        save_df(df, f"{cfg.out}/probabilistic_{i}_sparse.jsonl")

    for i in range(0, 110, 10):
        df['p_random'] = df.progress_apply(lambda r: probabilistic_retriever(con, r, i, K, fill='random'), axis=1)
        df['retrieved'] = df.progress_apply(lambda r: get_paragraphs(con, r['p_random'], K), axis=1)
        save_df(df, f"{cfg.out}/probabilistic_{i}_random.jsonl")

    raise Exception('Done')

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



