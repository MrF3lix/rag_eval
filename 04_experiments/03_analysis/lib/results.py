import os
import pandas as pd
import numpy as np
from prettytable import PrettyTable
import jax
import numpyro
import numpyro.distributions as dist
from sklearn.metrics import cohen_kappa_score

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

def is_retrieval_successful(row):
    ret_set = set([f"{r['document_id']}:{r['index']}" for r in row['retrieved']])
    ref_set = set([f"{r['document_id']}:{r['index']}" for r in row['reference']])

    return ref_set <= ret_set

def is_retrieval_partial_row(row):
    ref_set = set([f"{r['document_id']}:{r['index']}" for r in row['retrieved']])
    ret_set = set([f"{r['document_id']}:{r['index']}" for r in row['reference']])

    return bool(ref_set & ret_set)

def set_retriever_success(row):
    successful = is_retrieval_successful(row)
    partial = is_retrieval_partial_row(row)

    if successful == True:
        return 'SUCCESS'
    if partial == True:
        return 'PARTIAL'
    else:
        return 'FAILED'

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

def load_result_file(path, filtered = False, allowed_ids = []):
    df = pd.read_json(path, orient='records')
    df['abstain'] = df.apply(set_abstain, axis=1)
    df['retriever_success'] = df.apply(is_retrieval_successful, axis=1)
    df['retriever_result'] = df.apply(set_retriever_success, axis=1)
    df['generator_success'] = df.apply(set_generator_success, axis=1)
    df['task_success'] = df.apply(set_task_success, axis=1)

    if filtered == True:
        df = df.loc[(df['id'].isin(allowed_ids))]

    return df

def get_merged_ids(dfs, col='retriever_success', merge_operation = 'AND'):
    merged = dfs[0].merge(dfs[1], on="id", suffixes=("_a", "_b"))
    
    # Both Retrievers Agree
    if merge_operation == 'AND':
        matched = merged[
            merged[f"{col}_a"] == merged[f"{col}_b"]
        ]['id'].to_list()
    # Retrievers have different answers
    elif merge_operation == 'NAND':
        matched = merged[
            merged[f"{col}_a"] != merged[f"{col}_b"]
        ]['id'].to_list()
    # No Filtering
    elif merge_operation == 'NONE':
        return [], np.nan, np.nan
    
    pa = len(matched)/len(merged)*100
    ck = cohen_kappa_score(merged[f"{col}_a"], merged[f"{col}_b"])

    return matched, pa, ck

def collect_results(input, allowed_ids=[]):
    results = {}

    for id in os.listdir(input):
        try:
            info = id.split('_')
            data = info[0]
            dataset = info[2]
            retriever = info[3]
            generator = info[4]

            df = load_result_file(f"{input}/{id}/results.json", len(allowed_ids) > 0, allowed_ids)

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
        "a": [],
        "b": [],
        "f": [],
        "p_a1_r1": [],
        "p_a1_r0": []
    }
    for experiment, df in results.items():
        info = experiment.split('_')

        data['name'].append(experiment)
        data['retriever_success'].append(df['retriever_success'].mean())
        data['generator_success'].append(df['generator_success'].mean())
        data['task_success'].append(df['task_success'].mean())
        data['abstain'].append(df['abstain'].mean())

        data['retriever_strategy'].append(info[1])

        data['p_a1_r1'].append(df.loc[df['retriever_success']]['abstain'].mean())
        data['p_a1_r0'].append(df.loc[~df['retriever_success']]['abstain'].mean())

        x_obs = np.arange(0, 1.01, 0.1)
        samples = get_f_estimates(df, x_obs)

        a = df.loc[df['retriever_success'] == True]['task_success'].mean()
        data['a'].append((a, samples['a'].mean(), samples['a'].std()))
        b = df.loc[df['retriever_success'] == False]['task_success'].mean()
        data['b'].append((b, samples['b'].mean(), samples['b'].std()))
        data['f'].append((samples['f'].mean(), samples['f'].std()))

    df = pd.DataFrame(data)

    df['retriever_strategy'] = pd.Categorical(df['retriever_strategy'], categories=["empty", "oracle", "sparse", "dense", "hybrid"], ordered=True)
    df = df.sort_values('retriever_strategy')

    return results, df


def get_id_from_criteria(dfs, col='retriever_result', criteria_a='PARTIAL', criteria_b='PARTIAL'):
    merged = dfs[0].merge(dfs[1], on="id", suffixes=("_a", "_b"))

    matched_id = merged[
        (merged[f"{col}_a"] == criteria_a) &
        (merged[f"{col}_b"] == criteria_b)
    ]['id'].to_list()

    return matched_id

def load_allowed_ids(input, col='retriever_result', criteria_a='PARTIAL', criteria_b='PARTIAL'):
    dfs = []
    for id in os.listdir(input):
        try:
            dfs.append(load_result_file(f"{input}/{id}/results.json"))
        except Exception as e:
            print(e)
            print(f'Failed to load: {input}/{id}/results.json')
            continue
    
    return get_id_from_criteria(dfs, col, criteria_a, criteria_b)


def load_results_from_allowed_is(observations, key, allowed_ids):
    results = {}

    df_base, res = collect_results(observations[key]['base'], allowed_ids)
    results[f"{key}_base"] = res

    try:
        df_switched, res = collect_results(observations[key]['switched'], allowed_ids)
        results[f"{key}_switched"] = res
    except:
        return

    return results

def load_results(observations, merge_operation='AND', second_key='switched'):
    results = {}

    for key in observations.keys():
        input = observations[key]['base']

        dfs = []
        for id in os.listdir(input):
            try:
                dfs.append(load_result_file(f"{input}/{id}/results.json"))
            except Exception as e:
                print(e)
                print(f'Failed to load: {input}/{id}/results.json')
                continue

        allowed_ids, pa, ck = get_merged_ids(dfs, merge_operation=merge_operation)

        results[key] = {
            'pa': pa,
            'ck': ck,
            'n': len(allowed_ids) if len(allowed_ids)>0 else len(dfs[0])
        }

        df_base, res = collect_results(observations[key]['base'], allowed_ids)
        results[f"{key}_base"] = res

        try:
            df_switched, res = collect_results(observations[key][second_key], allowed_ids)
            results[f"{key}_{second_key}"] = res
        except:
            continue

    return results

def experiment_row(result, switched=False):
    info = result['name'].split('_')

    a_est = result['a'][1]
    a_std = result['a'][2]

    b_est = result['b'][1]
    b_std = result['b'][2]

    return [
        info[0],
        info[2],
        result['retriever_strategy'],
        switched,
        f"{result['retriever_success']:.2f}",
        f"{result['abstain']:.2f}",
        f"{result['task_success']:.4f}",
        f"{result['generator_success']:.4f}",
        f"{result['p_a1_r1']:.4f}",
        f"{result['p_a1_r0']:.4f}",
        f"{a_est:.4f} ±{a_std:.4f}",
        f"{b_est:.4f} ±{b_std:.4f}",
    ]


def print_dataset_results(dataset, results):
    t = PrettyTable(["Dataset", "Generator", "Retriever", "Switched", "P(R)", "P(A)", "P(T)", "P(G)", "P(A=1|R=1)", "P(A=1|R=0)", "P(T=1|R=1)", "P(T=1|R=0)"])

    df_base = results[f"{dataset}_base"]
    df_switched = results[f"{dataset}_switched"]

    for idx, row in df_base.iterrows():
        t.add_row(experiment_row(row, switched=False))
        try:
            t.add_row(experiment_row(df_switched.loc[df_switched['retriever_strategy'].astype(str) == str(row['retriever_strategy'])].iloc[0], switched=True))
        except:
            continue

    return t

def print_results(observations, results):
    t = PrettyTable(["Dataset", "Generator", "Retriever", "Switched", "P(R)", "P(A)", "P(T)", "P(G)", "P(A=1|R=1)", "P(A=1|R=0)", "P(T=1|R=1)", "P(T=1|R=0)"])

    for dataset in observations.keys():
        df_base = results[f"{dataset}_base"]
        df_switched = results[f"{dataset}_switched"]

        for idx, row in df_base.iterrows():
            t.add_row(experiment_row(row, switched=False))
            try:
                t.add_row(experiment_row(df_switched.loc[df_switched['retriever_strategy'].astype(str) == str(row['retriever_strategy'])].iloc[0], switched=True))
            except:
                continue

    return t