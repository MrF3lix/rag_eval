import { createContext } from "react";


export const GraphContext = createContext({
  successRates: {},
  setSuccessRates: () => {},
  conditionalProbabilities: {},
  setConditionalProbabilities: () => {}
});

export const parseLabel = (label) => {
  // Split node name from binary value, e.g. "q1" -> ["q", "1"]
  const match = label.match(/(.+)([01])$/);

  console.log(match)

  if (!match) {
    throw new Error(`Invalid label: ${label}`);
  }
  return match[1]; // just the node name
}

export const inferDependencyGraph = (conditionalProbs) => {
  const graph = {};

  for (const [node, config] of Object.entries(conditionalProbs)) {
    const p = config.p;
    if (typeof p === "number") {
      graph[node] = [];
    } else {
      const deps = new Set();
      console.log({p})

      for (const condKey of Object.keys(p)) {

        const labels = condKey.split("_");
        console.log(condKey, labels)

        if(!Array.isArray(labels)) {
          for (const label of labels) {
            console.log('label', label)
            deps.add(parseLabel(label));
          }
        }
      }
      graph[node] = Array.from(deps).sort();
    }
  }

  return graph;
}