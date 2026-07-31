import jax
import logging
import os
import numpyro
import duckdb
import logging
import argparse
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpyro.distributions as dist
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

def get_retrieval_success(row):
    ret_set = set([f"{r['document_id']}:{r['index']}" for r in row['retrieved']])
    ref_set = set([f"{r['document_id']}:{r['index']}" for r in row['reference']])

    return set(ref_set) <= set(ret_set)

def set_generator_success(row):
    if row['retriever_success'] == True:
        return row['correct_answer']
    else:
        return (row['generated_answer'] == 'I DO NOT KNOW') or (row['generated_answer'] == 'NOT ENOUGH INFO')
    
def set_task_success(row):
    return row['generated_answer'] == row['answer']
    
def set_abstain(row):
    a = row['generated_answer']
    return a == 'I DO NOT KNOW' or a == 'NOT ENOUGH INFO'

def collect_results(cfg, input):
    results = {}

    for id in os.listdir(input):
        print(id)

        try:
            info = id.split('_')
            data = info[0]
            dataset = info[2]
            retriever = info[3]
            generator = info[4]

            df = pd.read_json(f"{input}/{id}/results.json", orient='records')

            df['abstain'] = df.apply(set_abstain, axis=1)
            df['retriever_success'] = df.apply(get_retrieval_success, axis=1)
            df['generator_success'] = df.apply(set_generator_success, axis=1)
            df['task_success'] = df.apply(set_task_success, axis=1)

            results[f'{dataset}_{retriever}_{generator}'] = df
        except:
            continue

    data = {
        "name": [],
        "retriever_strategy": [],
        "retriever_success": [],
        "generator_success": [],
        "task_success": [],
        "abstain": [],
        "estimates": []
    }
    for experiment, df in results.items():
        info = experiment.split('_')

        data['name'].append(experiment)
        data['retriever_success'].append(df['retriever_success'].mean())
        data['generator_success'].append(df['generator_success'].mean())
        data['task_success'].append(df['task_success'].mean())
        data['abstain'].append(df['abstain'].mean())

        data['retriever_strategy'].append(info[1])


        a = df.loc[df['retriever_success'] == True]['task_success'].mean()
        b = df.loc[df['retriever_success'] == False]['task_success'].mean()

        data['estimates'].append((a,b))

    df = pd.DataFrame(data)
    return results, df

    

def get_prior_distributions(df):
    p_t1_r1 =  df.loc[(df['task_success'] == True) & (df['retriever_success'] == True)].shape[0]
    p_t1_r0 =  df.loc[(df['task_success'] == True) & (df['retriever_success'] == False)].shape[0]

    p_t0_r1 =  df.loc[(df['task_success'] == False) & (df['retriever_success'] == True)].shape[0]
    p_t0_r0 =  df.loc[(df['task_success'] == False) & (df['retriever_success'] == False)].shape[0]

    # P(T=1|R=1) ~ Beta(1+Count(T=1 and R=1), 1+Count(T=0 and R=1))
    a_p = dist.Beta(p_t1_r1+1, p_t0_r1+1)

    # P(T=1|R=0) ~ Beta(1+Count(T=1 and R=0), 1+Count(T=0 and R=0))
    b_p = dist.Beta(p_t1_r0+1, p_t0_r0+1)

    return a_p, b_p

def get_f_estimates(df, x_obs):
    a_p, b_p = get_prior_distributions(df)

    def model_fn(x_obs):
        a = numpyro.sample('a', a_p)
        b = numpyro.sample('b', b_p)

        numpyro.deterministic('f', (((a-b) * x_obs) + b))
            
    numpyro.set_host_device_count(10)
    RANDOM_SEED = 0xdeadbeef

    sampler = numpyro.infer.Predictive(model_fn, num_samples=10_000, params={'x_obs': x_obs})
    samples = sampler(jax.random.PRNGKey(RANDOM_SEED), x_obs)

    return samples

def task_success_estimates(cfg, results, report_path):
    fig, axs = plt.subplots(1, len(results.keys()), figsize=(18, 3))
    i = 0
    for experiment, df_r in results.items():
        ax = axs[i]

        a_p, b_p = get_prior_distributions(df_r)
        ax.set_title(experiment)

        ax.hist(a_p.sample(jax.random.PRNGKey(0), (1000,)), alpha=0.5, label="a")
        ax.hist(b_p.sample(jax.random.PRNGKey(0), (1000,)), alpha=0.5, label="b")
        ax.legend()
        ax.grid()

        i += 1

    plt.tight_layout()
    plt.savefig(f"{report_path}/task_success_estimates.png")


def plot_base(df, results, type, ax, show_estimates=True):
    x_obs = np.arange(0, 1.01, 0.1)

    df_b = df.loc[df['retriever_strategy'] == type]
    name = df_b['name'].iloc[0]
    df_exp = results[name]

    ax.plot(df_exp['retriever_success'].mean(), df_exp['task_success'].mean(), marker='x', label=type)
    color = plt.gca().lines[-1].get_color()

    if show_estimates:
        samples = get_f_estimates(df_exp, x_obs)
        f_samples = samples['f'] 
        f_mean = f_samples.mean(axis=0)

        p5  = np.percentile(f_samples, 5,  axis=0)
        p25 = np.percentile(f_samples, 25, axis=0)
        p75 = np.percentile(f_samples, 75, axis=0)
        p95 = np.percentile(f_samples, 95, axis=0)

        ax.fill_between(x_obs, p5, p95, color=color, alpha=0.15)
        ax.fill_between(x_obs, p25, p75, color=color, alpha=0.30)

        ax.plot(x_obs, f_mean, color=color)


def task_retrieval_succcess(cfg, df, results, report_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    plot_base(df, results, 'dense', ax)
    plot_base(df, results, 'sparse', ax)
    plot_base(df, results, 'hybrid', ax)
    plot_base(df, results, 'oracle', ax, False)
    plot_base(df, results, 'empty', ax, False)

    plt.xlabel("Retriever Success Rate")
    plt.ylabel("Task Success Rate") 
    plt.title(cfg.plot.title)

    plt.xlim(-0.05, 1.05)
    plt.ylim(0, 1)

    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{report_path}/task_retrieval_success.png")


def create_plots(cfg, input, report_path):

    results, df = collect_results(cfg, input)

    task_success_estimates(cfg, results, report_path)
    task_retrieval_succcess(cfg, df, results, report_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config/base.yaml"
    )
    parser.add_argument(
        "--input", type=str, help="Path to the results file", default="results/"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    now = datetime.today().strftime('%Y-%m-%d_%H-%M')
    report_path = f"{args.input}/{now}_plots"
    Path(report_path).mkdir(parents=True, exist_ok=True)
    logger.debug(f'Started {now}')

    create_plots(cfg, args.input, report_path)

main()



