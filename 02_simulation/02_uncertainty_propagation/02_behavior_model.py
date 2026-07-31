import jax
import numpyro
import numpy as np
import pandas as pd
import numpyro.distributions as dist
from collections import Counter
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

class SimulateBehaviorModel():
    def __init__(self):
        pass

    def load_conditionals(self, df):
        a_r, b_r = get_tp_fn(df['retriever_success'])

        a_a_r1, b_a_r1 = get_tp_fn(df.loc[df['retriever_success'] == True]['abstain'])
        a_a_r0, b_a_r0 = get_tp_fn(df.loc[df['retriever_success'] == False]['abstain'])

        a_t_r1_a1, b_t_r1_a1 = get_tp_fn(df.loc[(df['retriever_success'] == True)  & (df['abstain'] == True)]['task_success'])
        a_t_r1_a0, b_t_r1_a0 = get_tp_fn(df.loc[(df['retriever_success'] == True)  & (df['abstain'] == False)]['task_success'])
        a_t_r0_a1, b_t_r0_a1 = get_tp_fn(df.loc[(df['retriever_success'] == False) & (df['abstain'] == True)]['task_success'])
        a_t_r0_a0, b_t_r0_a0 = get_tp_fn(df.loc[(df['retriever_success'] == False) & (df['abstain'] == False)]['task_success'])

        return {
            'r': {
                'b': dist.Beta(a_r, b_r)
            },
            'a': {
                'b': {
                    'r1': dist.Beta(a_a_r1, b_a_r1),
                    'r0': dist.Beta(a_a_r0, b_a_r0)
                }
            },
            't': {
                'b': {
                    'r1_a1': dist.Beta(a_t_r1_a1, b_t_r1_a1),
                    'r1_a0': dist.Beta(a_t_r1_a0, b_t_r1_a0),
                    'r0_a1': dist.Beta(a_t_r0_a1, b_t_r0_a1),
                    'r0_a0': dist.Beta(a_t_r0_a0, b_t_r0_a0)
                }
            }
        }

    def compute_uncertainty(self, df, num_samples=10_000):
        def compute_a_success(r, a_r0, a_r1):
            return a_r0 * (1 - r) + a_r1 * r

        def compute_t_success(r, a_r0, a_r1, t_r1_a1, t_r1_a0, t_r0_a1, t_r0_a0):
            t_r1 = t_r1_a1 * a_r1 + t_r1_a0 * (1 - a_r1)
            t_r0 = t_r0_a1 * a_r0 + t_r0_a0 * (1 - a_r0)

            return t_r1 * r + t_r0 * (1 - r)

        def compute_g_success(r, a_r0, a_r1, t_r1_a0):
            # G      = (R=0 AND A=1) OR (R=1 AND A=0 AND T=1)
            # P(G=1) = P(R=0) * P(A=1|R=0) + P(R=1) * P(A=0|R=1) * P(T=1|R=1,A=0)
            return (1 - r) * a_r0 + r * (1 - a_r1) * t_r1_a0

        def model_fn(conditionals):
            r = numpyro.sample("r", conditionals['r']['b'])

            a_r1 = numpyro.sample("a_r1", conditionals['a']['b']['r1'])
            a_r0 = numpyro.sample("a_r0", conditionals['a']['b']['r0'])

            numpyro.deterministic("a", compute_a_success(r, a_r0, a_r1))

            t_r1_a1 = numpyro.sample("t_r1_a1", conditionals['t']['b']['r1_a1'])
            t_r1_a0 = numpyro.sample("t_r1_a0", conditionals['t']['b']['r1_a0'])
            t_r0_a1 = numpyro.sample("t_r0_a1", conditionals['t']['b']['r0_a1'])
            t_r0_a0 = numpyro.sample("t_r0_a0", conditionals['t']['b']['r0_a0'])

            numpyro.deterministic("t", compute_t_success(r, a_r0, a_r1, t_r1_a1, t_r1_a0, t_r0_a1, t_r0_a0))
            numpyro.deterministic("g", compute_g_success(r, a_r0, a_r1, t_r1_a0))

        conditionals = self.load_conditionals(df)
        sampler = infer.Predictive(model_fn, num_samples=num_samples, params={'conditionals': conditionals})
        samples = sampler(jax.random.PRNGKey(RANDOM_SEED), conditionals)

        res = {}
        for k in samples.keys():
            res[k] = {
                'mean': samples[k].mean().item(),
                'std': samples[k].std().item()
            }

        return res, samples