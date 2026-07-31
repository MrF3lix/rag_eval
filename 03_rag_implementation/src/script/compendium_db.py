import duckdb

SOURCE_DB = 'data/compendium/catechism_english.jsonl'
TARGET_DB = 'data/compendium/catechism.duckdb'

con = duckdb.connect(TARGET_DB)

con.sql("DROP TABLE IF EXISTS paragraph")

con.execute(f"CREATE TABLE paragraph (document_id VARCHAR, title VARCHAR, global_id BIGINT, index INTEGER, text VARCHAR);")

con.execute(f"""
INSERT INTO paragraph
SELECT 
    num AS document_id,
    'TITLE' AS title,
    num AS global_id,
    num AS index,
    text
FROM '{SOURCE_DB}'
""")
