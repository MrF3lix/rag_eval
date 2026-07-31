import json
from pathlib import Path
from retriv import SparseRetriever, set_base_path

from .base_index import BaseIndex

class SparseIndex(BaseIndex):
    def __init__(self, cfg):
        self.cfg = cfg

        p = Path(cfg.index.sparse.path)
        set_base_path(str(p.parent))

        self.sr = SparseRetriever(
            index_name=p.name,
            model="bm25",
            min_df=1,
            tokenizer="whitespace",
            stemmer="english",
            stopwords="english",
            do_lowercasing=True,
            do_ampersand_normalization=True,
            do_special_chars_normalization=True,
            do_acronyms_normalization=True,
            do_punctuation_removal=True,
        )

    def add_paragraphs(self, batch):
        with open(self.cfg.index.sparse.temp_file, "a", encoding="utf-8") as f:
            for global_id, text in batch:
                record = {
                    "global_id": global_id,
                    "text": text
                }
                f.write(json.dumps(record) + "\n")
            f.flush()

    def save_index(self):
        self.sr.index_file(
            path=self.cfg.index.sparse.temp_file,
            show_progress=True,
            callback=lambda doc: {
                "id": doc["global_id"],
                "text": doc["text"],          
            }
        )