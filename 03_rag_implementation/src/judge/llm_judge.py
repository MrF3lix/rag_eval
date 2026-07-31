from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam
from retriever import Query
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .base_judge import BaseJudge
from llm_backend import load_llm_backend

class LLMJudge(BaseJudge):
    def __init__(self, cfg):
        self.cfg = cfg

        self.env = Environment(
            loader=FileSystemLoader('prompts'),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )

        self.system_template = self.env.get_template(cfg.judge.system_prompt_template)
        self.user_template = self.env.get_template(cfg.judge.user_prompt_template)

        self.llm = load_llm_backend(
            base_url=self.cfg.judge.base_url,
            api_key=self.cfg.judge.api_key
        )

    
    def evaluate(self, query: Query) -> Query:
        query.retrieved_correct_document = self.retrieved_correct_document(query)
        query.retrieved_correct_paragraph = self.retrieved_correct_paragraph(query)
        query = self.generated_answer_correct(query)

        return query
    
    def generated_answer_correct(self, query: Query):
        self.client = OpenAI(base_url=self.cfg.judge.base_url, api_key=self.cfg.judge.api_key)


        response = self.llm.send_request(
            model=self.cfg.judge.model,
            messages=[
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=self.system_template.render()
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=self.user_template.render(
                        question=query.input,
                        reference=query.answer,
                        hypothesis=query.generated_answer,
                    )
                )
            ],
            temperature=self.cfg.judge.temperature,
            max_tokens=self.cfg.judge.max_tokens,
            thinking=self.cfg.judge.thinking
        )

        raw = response.choices[0].message.content
        response = raw.split('</think>')[-1]
        answer = response.split('\n')
        answer = [True if 'YES' in a else False for a in answer]

        # query.use_llm_judge = True
        # query.llm_judge_answer = response
        query.task_success = all(answer[i] for i in self.cfg.judge.required_items)
        query.task_success_score = sum(answer)
        # query.generator_success = all(answer[i] for i in self.cfg.judge.required_items)

        # if 'success_answer' in self.cfg.judge.keys():
        #     query.generator_success = all(answer[i] == False for i in self.cfg.judge.required_items)
            # TODO: define when generator_success differs from task success

        query.detailed_generator_eval = {}
        query.detailed_generator_eval['answers'] = answer
        query.detailed_generator_eval['raw'] = raw
        # for dim in self.cfg.judge.dimensions.keys():
        #     total = 0
        #     indices = self.cfg.judge.dimensions[dim]
        #     for i in indices:
        #         total += answer[int(i)-1]

        #     query.detailed_generator_eval.setdefault(f"{dim}_accuracy", 0.0)
        #     query.detailed_generator_eval[f"{dim}_accuracy"] = total / len(indices)
        #     query.detailed_generator_eval.setdefault(dim, True if total == len(indices) else False)
            
        return query
