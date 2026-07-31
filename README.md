# On the Limits of RAG Evaluation
*Modeling Error and Uncertainty Propagation in Retrieval-Augmented Generation*

[Read Thesis](./2026_MT__RAG_Eval__Felix_Saaro.pdf)

## Abstract

Master's thesis exploring why Retrieval-Augmented Generation (RAG) systems are difficult to evaluate. Introduces a System model and a Behavior model that formalize a RAG pipeline as a joint distribution over query, retrieval, and generation success, distinguishing *task success* (did the system answer correctly?) from *generator success* (did it behave as expected, including abstaining when retrieval fails?). Extends these models with a simulation-based approach to study how error and uncertainty in individual component estimates propagate through the pipeline.

Validated empirically across 27 RAG configurations (3 retrievers × 3 generators) on four QA/fact-verification datasets (FEVER, NQ, HotpotQA, C4). Finds that evaluation results do not generalize across configurations, even when a metric like retrieval success is held fixed, because retrieval and generation are coupled through abstention behavior. Also shows that uncalibrated LLM-as-judge evaluation introduces further, unquantified disagreement between judges.

Includes an [interactive tool](https://system-failure-simulator.vercel.app/) [[Source Code](./02_simulation/02_error_propagation/README.md)] for exploring error propagation in arbitrary acyclic pipeline systems.

## Supplementary Material

### Simulation

- [Error Propagation](./02_simulation/02_error_propagation/README.md)
- [Uncertainty Propagation](./02_simulation/02_uncertainty_propagation/README.md)

### RAG Implementation

- [Reference Implementation](./03_rag_implementation/README.md)

### Experiments

- Configurations: `04_experiments/01_config`
- Data: `04_experiments/02_data` (excluded on github, for access please send request by mail)
- Analysis including Scripts for Table and Plots: `04_experiments/03_analysis`