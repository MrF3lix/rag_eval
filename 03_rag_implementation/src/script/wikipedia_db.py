import duckdb

SOURCE_DB = 'data/wikipedia_kilt.duckdb'
TARGET_DB = 'data/all.duckdb'

con = duckdb.connect(TARGET_DB)

con.sql("DROP TABLE IF EXISTS wiki")
con.sql("DROP TABLE IF EXISTS paragraph")
con.sql("DROP SEQUENCE IF EXISTS paragraph_id")

con.execute(f"""
    ATTACH '{SOURCE_DB}' AS src;
    
    CREATE TABLE wiki AS
    SELECT *
    FROM src.wiki;
""")

con.sql("CREATE TABLE paragraph (document_id VARCHAR, wikipedia_title VARCHAR, global_id BIGINT, index INTEGER, text VARCHAR);")
con.sql("CREATE SEQUENCE paragraph_id START 1;")

con.execute("""
INSERT INTO paragraph
SELECT
    wikipedia_id as document_id,
    wikipedia_title,
    nextval('paragraph_id') as global_id,
    idx AS index,
    paragraph AS text
FROM wiki, UNNEST(text.paragraph) WITH ORDINALITY AS t(paragraph, idx);
""")