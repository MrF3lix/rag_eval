## RAG Implementation

Refernce implementation for the RAG system and the evaluation components used to run all experiments in this thesis.
The pipeline is built around a shared `Query` data model that is passed through and extended by each task: `Retriever` attaches the retrieved paragraphs, `Generator` attaches the generated answer, and `Judge` attaches the evaluation metrics. Each task is implemented against an abstract base class (`BaseRetriever`, `BaseGenerator`, `BaseJudge`), so new retrieval strategies, generation backends, or evaluation metrics can be added without changing the rest of the pipeline.

A run consists of five stages — `init`, `precompute`, `retrieve`, `generate`, `judge` — configured via YAML and runnable either end-to-end or independently, so that, for example, retrieval can be precomputed once at `k=100` and reused across experiments with smaller context sizes. Three retriever strategies (sparse/BM25, dense/embedding-based, and hybrid via Reciprocal Rank Fusion) and multiple LLM backends (OpenAI-compatible APIs, Ollama, vLLM) are supported out of the box.

### Prerequisits

1. Clone the repository
2. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
3. Install dependencies
```bash
uv sync
```

### Getting Started

1. Create a configuration to initialize the knowledge base (Examples are in the `04_experiments/01_config` directory).
Example for a knowledge base for the fever dataset with dense retrieval in mind.

```yaml
documents:
  subset_size: 10000
  target: <target_folder>/kilt_fever_train.jsonl
  dataset: fever
knowledge_base:
  source: data/wikipedia_kilt.duckdb
  target: <target_folder>/kilt_wiki.duckdb
  use_subset: True
embedder:
  model: jinaai/jina-embeddings-v3
  task: text-matching
  query_task: retrieval.query
index:
  dim: 1024
  batch_size: 25
  dense:
    path: <target_folder>/kilt_wiki.index
```

2. Run the knowledgebase initialization

```bash
uv run ./src/01_init_knowledge_base.py --config <path_to_config>

```

3. Run the pipeline

```bash
uv run ./src/02_run_pipeline.py --config <path_to_config>
```

4. Run individual tasks

```bash
uv run ./src/03_precompute_retriever.py --config <path_to_config>/precompute.yaml
uv run ./src/04_retrieve_from_precomputed.py --config <path_to_config>/retrieve.yaml
uv run ./src/05_generator.py --config <path_to_config>/generator.yaml" --output <path_to_output>
uv run ./src/06_judge.py --config <path_to_config>/judge.yaml"
```

### Data Model

The shared data model definition.
```python
class Paragraph():
    document_id: int
    index: int
    global_id: Optional[int] = None
    text: Optional[str] = None

class Query():
    model_config = ConfigDict(strict=True)
    id: str
    input: str
    answer: Optional[str] = None
    reference: list[Paragraph] = []
    retrieved: list[Paragraph] = []
    generated_answer: Optional[str] = None

    retriever_success: Optional[bool] = None
    abstain: Optional[bool] = None
    task_success: Optional[bool] = None
    generator_success: Optional[bool] = None
```