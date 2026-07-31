from typing import Optional
from pydantic import BaseModel, ConfigDict


class Paragraph(BaseModel):
    document_id: int
    index: int
    global_id: Optional[int] = None
    text: Optional[str] = None

class Query(BaseModel):
    model_config = ConfigDict(strict=True)
    id: str
    input: str
    answer: Optional[str] = None
    reference: list[Paragraph] = []
    retrieved: list[Paragraph] = []
    generated_answer: Optional[str] = None
    retriever_success: Optional[bool] = None
    task_success: bool = None
    task_success_score: int = 0

    target_answer: Optional[str] = None
    use_llm_judge: bool = False
    llm_judge_answer: Optional[str] = False
    generator_success: bool = None
    is_query_correct: bool = True
    is_retrieved_adjusted: bool = False
    retrieved_correct_document: Optional[bool] = None
    retrieved_correct_paragraph: Optional[bool] = None
    detailed_generator_eval: Optional[dict] = None

    def compute_result(self):
        return {
            'id': self.id,
            'input': self.input,
            'reference': list(map(lambda r: r.model_dump(), self.reference)),
            'retrieved': list(map(lambda r: r.model_dump(), self.retrieved)),
            'answer': self.answer,
            'generated_answer': self.generated_answer,
            'retriever_success': is_retrieval_successful(self),
            'task_success': set_task_success(self),
            'task_success_score': self.task_success_score,
            'generator_success': set_generator_success(self),
            'abstain': set_abstain(self),
            'retriever_recall@10': recall_at_k(self, 10),
            'detailed_generator_eval': self.detailed_generator_eval
        }
    

def is_retrieval_successful(query: Query):
    ret_set = set([f"{r.document_id}:{r.index}" for r in query.retrieved])
    ref_set = set([f"{r.document_id}:{r.index}" for r in query.reference])

    return ref_set <= ret_set

def set_generator_success(query: Query):
    if query.retriever_success == True:
        return query.generator_success
    else:
        return set_abstain(query)
    
def set_task_success(query: Query):
    return query.generated_answer == query.answer
    
def set_abstain(query: Query):
    return (query.generated_answer == 'I DO NOT KNOW') or (query.generated_answer == 'NOT ENOUGH INFO')

def recall_at_k(query: Query, k: int = 10) -> float:
    retrieved = [p.global_id for p in query.retrieved]
    references = set([p.global_id for p in query.reference])

    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in references)
    return hits / len(references) if references else 0.0