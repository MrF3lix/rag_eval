
import jax
import numpy as np
import pandas as pd
from collections import Counter
import numpyro.distributions as dist
import numpyro
from numpyro import infer

numpyro.set_host_device_count(10)
RANDOM_SEED = 0xdeadbeef

def divide(a, b):
    if b == 0:
        return 0
    return a/b

def get_tp_fn(col):
    counts = Counter(np.array(col, dtype=int))
    return [counts[1]+1, counts[0]+1]

def get_success_rate(col):
    tp, fn = get_tp_fn(col)

    return divide(tp, (tp + fn))

class Simulate():
    def __init__(self):
        pass

    def load_conditionals(self, df):
        a_q, b_q = get_tp_fn(df['correct_query'])

        a_r_q1, b_r_q1 =  get_tp_fn(df.loc[df['correct_query'] == True]['retriever_success'])
        a_r_q0, b_r_q0 =  get_tp_fn(df.loc[df['correct_query'] == False]['retriever_success'])

        a_g_r1_q1, b_g_r1_q1 =  get_tp_fn(df.loc[(df['correct_query'] == True)  & (df['retriever_success'] == True)]['task_success'])
        a_g_r1_q0, b_g_r1_q0 =  get_tp_fn(df.loc[(df['correct_query'] == False)  & (df['retriever_success'] == True)]['task_success'])
        a_g_r0_q1, b_g_r0_q1 =  get_tp_fn(df.loc[(df['correct_query'] == True)  & (df['retriever_success'] == False)]['task_success'])
        a_g_r0_q0, b_g_r0_q0 =  get_tp_fn(df.loc[(df['correct_query'] == False)  & (df['retriever_success'] == False)]['task_success'])


        return {
            'q': {
                'b': dist.Beta(a_q, b_q)
            },
            'r': {
                'b': {
                    'q1': dist.Beta(a_r_q1, b_r_q1),
                    'q0': dist.Beta(a_r_q0, b_r_q0)
                }
            },
            'g': {
                'b': {
                    'r1_q1': dist.Beta(a_g_r1_q1, b_g_r1_q1),
                    'r1_q0': dist.Beta(a_g_r1_q0, b_g_r1_q0),
                    'r0_q1': dist.Beta(a_g_r0_q1, b_g_r0_q1),
                    'r0_q0': dist.Beta(a_g_r0_q0, b_g_r0_q0)
                }
            }
        }
    
    def compute_uncertainty(self, df):
        def compute_r_success(q, r_q0, r_q1):
            return r_q0 * (1 - q) + r_q1 * q

        def compute_g_success(q, r_q0, r_q1, g_r1_q1, g_r1_q0, g_r0_q1, g_r0_q0):
            g_q1 = g_r1_q1 * r_q1 + g_r0_q1 * (1 - r_q1)
            g_q0 = g_r1_q0 * r_q0 + g_r0_q0 * (1 - r_q0)

            return  g_q1 * q + g_q0 * (1 - q)

        def model_fn(conditionals):
            q = numpyro.sample("q", conditionals['q']['b'])

            r_q1 = numpyro.sample("r_q1", conditionals['r']['b']['q1'])
            r_q0 = numpyro.sample("r_q0", conditionals['r']['b']['q0'])

            numpyro.deterministic("r", compute_r_success(q, r_q0, r_q1))

            g_r1_q1 = numpyro.sample("g_r1_q1", conditionals['g']['b']['r1_q1'])
            g_r1_q0 = numpyro.sample("g_r1_q0", conditionals['g']['b']['r1_q0'])
            g_r0_q1 = numpyro.sample("g_r0_q1", conditionals['g']['b']['r0_q1'])
            g_r0_q0 = numpyro.sample("g_r0_q0", conditionals['g']['b']['r0_q0'])

            numpyro.deterministic("g", compute_g_success(q, r_q0, r_q1, g_r1_q1, g_r1_q0, g_r0_q1, g_r0_q0))

        conditionals = self.load_conditionals(df)
        sampler = infer.Predictive(model_fn, num_samples=10_000, params={'conditionals': conditionals})
        samples = sampler(jax.random.PRNGKey(RANDOM_SEED), conditionals)


        res = {}
        for k in samples.keys():
            res[k] = {
                'mean': samples[k].mean().item(),
                'std': samples[k].std().item()
            }

        return res