from retriever import Query

from .base_judge import BaseJudge

class DefaultJudge(BaseJudge):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
    
    def evaluate(self, query: Query) -> Query:
        query.retrieved_correct_document = self.retrieved_correct_document(query)
        query.retrieved_correct_paragraph = self.retrieved_correct_paragraph(query)
        query.task_success = self.generated_answer_correct(query)


        if 'judge' in self.cfg.keys() and 'success_answer' in self.cfg.judge.keys():
            query.generator_success = query.generated_answer == self.cfg.judge.success_answer
        else:
            query.generator_success = query.task_success # TODO: define when generator_success differs from task success

        if query.target_answer != None:
            query.generator_success = query.generated_answer == query.target_answer


        return query
    
    def generated_answer_correct(self, query):
        return query.answer == query.generated_answer
