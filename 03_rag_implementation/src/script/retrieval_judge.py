import asyncio
import json
import pandas as pd
from scipy.stats import pearsonr
from tqdm.asyncio import tqdm as tqdm_async
from ollama import AsyncClient
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam
from jinja2 import Environment, FileSystemLoader, StrictUndefined


N = 10000
SEED = 42
NUM_PROC = 10
RES = 'results/2026-03-12_nq/qwen/2026-03-12_15-59_nq_dense_qwen/results.jsonl'
# MODEL = 'llama3.2'

MODEL = 'Qwen/Qwen3.5-9B'
URL = 'https://uigq0azsrs75om-8000.proxy.runpod.net/v1'
API_KEY = 'sk-uigq0azsrs75om'

async def retrieval_useful(row, system, user):
    client = AsyncOpenAI(
        base_url=URL,
        api_key=API_KEY,
    )

    model_response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            ChatCompletionSystemMessageParam(
                role="system",
                content=system.render()
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=user.render(
                    input=row['input'],
                    answer=row['generated_answer'],
                    retrieved=row['retrieved']
                )
            ),
        ],
        max_tokens=1,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    return model_response.choices[0].message.content == 'Yes'

async def main():

    env = Environment(
        loader=FileSystemLoader('prompts'),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )

    system_template = env.get_template('nq_judge_system.jinja')
    user_template = env.get_template('nq_judge_user.jinja')


    df = pd.read_json(RES, lines=True, orient='records')
    df = df.sample(N, random_state=SEED)

    sem = asyncio.Semaphore(NUM_PROC)

    async def worker(req):
        async with sem:
            return await retrieval_useful(req, system_template, user_template)

    tasks = [asyncio.create_task(worker(r)) for _, r in df.iterrows()]
    results = await tqdm_async.gather(*tasks, total=len(tasks))
    df['qa_retriever_success'] = results

    print(df[['correct_paragraph', 'qa_retriever_success']].corr(method='spearman'))
    print(df[['correct_paragraph', 'qa_retriever_success']].corr(method='pearson'))

asyncio.run(main())