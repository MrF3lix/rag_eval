## Error Propagation Simulation

An interactive tool for exploring error propagation in pipelined systems, built alongside this thesis to make the System model (Section 2.2.2) tangible and explorable beyond the fixed example used in the text.

The tool lets you construct any acyclic pipeline by adding nodes and connecting them, then specifying the conditional success probability of each node given the state of its inputs. Marginal success probabilities are recomputed automatically as you adjust these conditionals, making it possible to see, in real time, how an estimation error or improvement at one component propagates through the rest of the system.

While built for RAG pipelines (Query → Retriever → Generator), the tool is not specific to RAG: any system that can be described as a directed acyclic graph of binary success/failure components can be modeled with it.

**Try it here:** [Error Propagation Simulator](https://system-failure-simulator.vercel.app/)