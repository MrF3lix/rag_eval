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

def fill_random_context(con, row, input='sparse', cutoff=5):
    ref = list(set(row['ref']))
    ret_input = row[input][:cutoff]
    correct_documents = list(set(ref) & set(ret_input))

    random = get_random_documents(con, cutoff-len(correct_documents), ref)
    return correct_documents + random


def retrieve(cfg):
    K = cfg.K

    con = duckdb.connect(cfg.wiki_all)
    df = pd.read_json(cfg.precomputed, lines=True)
    # df = df.head(1)

    logger.info('Preprocess Precomputed Results')
    df['id'] = df['id'].astype(str)
    df['references'] = df['references'].progress_apply(lambda r: get_paragraph_with_global_id(con, r))
    df['ref'] = df['references'].apply(lambda r: [element['global_id'] for element in r])

    df['retrieved'] = df.progress_apply(lambda r: get_paragraphs(con, r['sparse'], K), axis=1)
    save_df(df, f"{cfg.out}/sparse.jsonl")
    df['retrieved'] = df.progress_apply(lambda r: get_paragraphs(con, r['dense'], K), axis=1)
    save_df(df, f"{cfg.out}/dense.jsonl")

    df['sparse_p'] = df.progress_apply(lambda r: fill_random_context(con, r, 'sparse', K), axis=1)
    df['retrieved'] = df.progress_apply(lambda r: get_paragraphs(con, r['sparse_p'], K), axis=1)
    save_df(df, f"{cfg.out}/sparse_random.jsonl")

    df['dense_p'] = df.progress_apply(lambda r: fill_random_context(con, r, 'dense', K), axis=1)
    df['retrieved'] = df.progress_apply(lambda r: get_paragraphs(con, r['dense_p'], K), axis=1)
    save_df(df, f"{cfg.out}/dense_random.jsonl")

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



