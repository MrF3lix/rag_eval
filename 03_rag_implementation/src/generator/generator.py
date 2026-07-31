import logging
import time
import random
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from generator import BaseGenerator
from retriever import Query
from llm_backend import load_llm_backend

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

class Generator(BaseGenerator):
    def __init__(self, cfg):
        self.cfg = cfg

        self.env = Environment(
            loader=FileSystemLoader('prompts'),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )

        self.system_template = self.env.get_template(cfg.generator.system_prompt_template)
        self.user_template = self.env.get_template(cfg.generator.user_prompt_template)

        self.llm = load_llm_backend(
            base_url=self.cfg.generator.base_url,
            api_key=self.cfg.generator.api_key
        )

    def generate(self, query: Query) -> Query:
        delay = .5
        response = None
        for attempts_left in range(6, -1, -1):
            try:
                response = self.llm.send_request(
                    model=self.cfg.generator.model,
                    messages=[
                        ChatCompletionSystemMessageParam(
                            role="system",
                            content=self.system_template.render()
                        ),
                        ChatCompletionUserMessageParam(
                            role="user",
                            content=self.user_template.render(
                                input=query.input,
                                retrieved=query.retrieved
                            )
                        )
                    ],
                    temperature=self.cfg.generator.temperature,
                    max_tokens=self.cfg.generator.max_tokens,
                    thinking=self.cfg.generator.thinking
                )
            except Exception as e:
                if attempts_left > 0:
                    jitter = random.random()*0.25
                    time.sleep(delay + jitter)
                    delay = min(delay * 2, 10)
                else:
                    print('FAILED REQUEST: ', query.id)
                    query.generated_answer = "FAILED"
            finally:
                if response is not None:
                    query.generated_answer = response.choices[0].message.content
                    break

        return query