import duckdb
from datasets import load_dataset
from tqdm import tqdm

from .base_kb import KnowledgeBase
from index import DenseIndex, SparseIndex

class WikiKnowledgeBase(KnowledgeBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.index = DenseIndex(cfg) if 'dense' in cfg.index else SparseIndex(cfg)
        self.con = duckdb.connect(cfg.knowledge_base.target)

    def init_database(self):
        self.con.sql("DROP TABLE IF EXISTS wiki")
        self.con.sql("DROP TABLE IF EXISTS paragraph")
        self.con.sql("DROP SEQUENCE IF EXISTS paragraph_id")

        self.init_wiki_table()

        self.con.sql("CREATE TABLE paragraph (document_id VARCHAR, title VARCHAR, global_id BIGINT, index INTEGER, text VARCHAR);")
        self.con.sql("CREATE SEQUENCE paragraph_id START 1;")

        self.con.execute("""
        INSERT INTO paragraph
        SELECT
            wikipedia_id as document_id,
            wikipedia_title as title,
            nextval('paragraph_id') as global_id,
            idx AS index,
            paragraph AS text
        FROM wiki, UNNEST(text.paragraph) WITH ORDINALITY AS t(paragraph, idx);
        """)

    def init_wiki_table(self):
        relevant_wiki_pages = self.select_subset()

        if 'use_subset' in self.cfg.knowledge_base.keys() and self.cfg.knowledge_base.use_subset == True:
            if 'unseen' in self.cfg.knowledge_base.keys() and self.cfg.knowledge_base.unseen == True:
                print('UNSEEN')
                self.con.execute(f"""
                    ATTACH '{self.cfg.knowledge_base.source}' AS src;
                    CREATE TABLE wiki AS
                    SELECT *
                    FROM src.wiki
                    WHERE wikipedia_id NOT IN ?
                    LIMIT ?;
                """,  [relevant_wiki_pages, self.cfg.documents.subset_size])
            else:
                self.con.execute(f"""
                    ATTACH '{self.cfg.knowledge_base.source}' AS src;
                    
                    CREATE TABLE wiki AS
                    SELECT *
                    FROM src.wiki
                    WHERE wikipedia_id in ?;
                """,  [relevant_wiki_pages])
        else:
            self.con.execute(f"""
                ATTACH '{self.cfg.knowledge_base.source}' AS src;
                
                CREATE TABLE wiki AS
                SELECT *
                FROM src.wiki;
            """)

    def select_subset(self):
        ds = load_dataset("facebook/kilt_tasks", name=self.cfg.documents.dataset)
        ds = ds['train'].filter(lambda row: (len(row['output'][0]['provenance']) > 0))

        subset = ds.select(range(self.cfg.documents.subset_size))
        if 'subset_size' in self.cfg.documents.keys():
            subset = ds.select(range(self.cfg.documents.subset_size))
        else:
            subset = ds

        relevant = subset.map(self.extract_rows)
        relevant.select_columns(['id', 'input', 'answer', 'references']).to_json(self.cfg.documents.target)

        document_ids = []
        for relevant in relevant['references']:
            document_ids.extend([p['document_id'] for p in relevant])

        return document_ids

    def extract_rows(self, row):
        references = []
        for ref in row['output'][0]['provenance']:
            references.append({
                'document_id': ref['wikipedia_id'],
                'index': ref['start_paragraph_id'] + 1,
                'global_id': None,
                'text': None
            })

        answer = row['output'][0]['answer']

        return {
            'answer': answer,
            'references': references,
        }

    def init_index(self):
        total = self.con.execute("SELECT count(*) FROM paragraph;").fetchall()[0][0]
        cursor = self.con.execute("SELECT global_id, text FROM paragraph")
        pbar = tqdm(total=total)
        while True:
            rows = cursor.fetchmany(self.cfg.index.batch_size)
            if not rows:
                break

            self.index.add_paragraphs(rows)
            pbar.update(self.cfg.index.batch_size)

        self.index.save_index()
