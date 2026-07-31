self.onmessage = function(e) {
  const { nodes, N, seed } = e.data;
  const result = simulate({ nodes, N, seed });
  
  self.postMessage(result);
};

// Utility: parse dependencies from keys like "r1_q0"
function parseDependencies(cpt) {
  const deps = new Set();
  for (const condKey of Object.keys(cpt)) {
    for (const label of condKey.split("_")) {
      deps.add(label[0]); // first char is variable name
    }
  }
  return deps;
}

// Simulate system for a single node
function simulateSystem(baseProbs, prevStates, N, rng) {
  const out = new Array(N).fill(false);
  for (const [condKey, p] of Object.entries(baseProbs)) {
    const labels = condKey.split("_");
    const mask = new Array(N).fill(true);

    labels.forEach(label => {
      const [varName, val] = [label[0], label[1]];
      for (let i = 0; i < N; i++) {
        mask[i] = mask[i] && (val === "1" ? prevStates[varName][i] : !prevStates[varName][i]);
      }
    });

    for (let i = 0; i < N; i++) {
      if (mask[i]) {
        out[i] = rng() < p;
      }
    }
  }
  return out;
}

// Main simulate function
function simulate({ nodes, N = 100000, seed = 0 }) {
  // Simple RNG: mulberry32
  function mulberry32(a) {
    return function() {
      let t = a += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  const rng = mulberry32(seed);

  const states = {};
  const remaining = new Set(Object.keys(nodes));

  while (remaining.size > 0) {
    for (const node of Array.from(remaining)) {
      const nodeDef = nodes[node];
      if (typeof nodeDef.p === "number") {
        // unconditional
        states[node] = Array.from({ length: N }, () => rng() < nodeDef.p);
        remaining.delete(node);
      } else {
        const deps = parseDependencies(nodeDef.p);
        if ([...deps].every(d => d in states)) {
          states[node] = simulateSystem(nodeDef.p, states, N, rng);
          remaining.delete(node);
        }
      }
    }
  }

  // Collect probabilities
  const results = {};
  for (const n of Object.keys(nodes)) {
    const arr = states[n];
    results[n] = arr.reduce((acc, val) => acc + (val ? 1 : 0), 0) / N;
  }
  return results;
}