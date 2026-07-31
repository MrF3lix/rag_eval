self.onmessage = function(e) {
  const { nodes, N, seed } = e.data;
  const result = simulate({ nodes, N, seed });
  
  self.postMessage(result);
};

// RNG utilities
function mulberry32(a) {
  return function() {
    let t = a += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Gamma and Beta samplers
function sampleGamma(k, theta, rng) {
  if (k < 1) {
    const u = rng();
    return sampleGamma(1 + k, theta, rng) * Math.pow(u, 1 / k);
  }
  const d = k - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x, v;
    do {
      const u1 = rng(), u2 = rng();
      const n = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); // Normal(0,1)
      x = n;
      v = Math.pow(1 + c * x, 3);
    } while (v <= 0);
    const u = rng();
    if (u < 1 - 0.0331 * Math.pow(x, 4)) return d * v * theta;
    if (Math.log(u) < 0.5 * Math.pow(x, 2) + d * (1 - v + Math.log(v)))
      return d * v * theta;
  }
}

function sampleBeta(alpha, beta, rng) {
  const x = sampleGamma(alpha, 1, rng);
  const y = sampleGamma(beta, 1, rng);
  return x / (x + y);
}

// Parse dependencies like before
function parseDependencies(cpt) {
  const deps = new Set();
  for (const condKey of Object.keys(cpt)) {
    for (const label of condKey.split("_")) deps.add(label[0]);
  }
  return deps;
}

// Simulate conditional system
function simulateSystem(baseProbs, prevStates, N, rng) {
  const out = new Array(N).fill(false);
  for (const [condKey, pDef] of Object.entries(baseProbs)) {
    // Get actual probability (number or Beta sample)
    let p;
    if (Array.isArray(pDef)) {
      const [a, b] = pDef;
      p = sampleBeta(a, b, rng);
    } else {
      p = pDef;
    }

    const labels = condKey.split("_");
    const mask = new Array(N).fill(true);

    for (const label of labels) {
      const [varName, val] = [label[0], label[1]];
      for (let i = 0; i < N; i++) {
        mask[i] = mask[i] && (val === "1" ? prevStates[varName][i] : !prevStates[varName][i]);
      }
    }

    for (let i = 0; i < N; i++) {
      if (mask[i]) out[i] = rng() < p;
    }
  }
  return out;
}

// Main simulation
function simulate({ nodes, N = 100000, seed = 0 }) {
  const rng = mulberry32(seed);
  const states = {};
  const remaining = new Set(Object.keys(nodes));

  while (remaining.size > 0) {
    for (const node of Array.from(remaining)) {
      const nodeDef = nodes[node];
      const pDef = nodeDef.p;

      if (typeof pDef === "number" || Array.isArray(pDef)) {
        // Unconditional
        let p;
        if (Array.isArray(pDef)) {
          const [a, b] = pDef;
          p = sampleBeta(a, b, rng);
        } else {
          p = pDef;
        }
        states[node] = Array.from({ length: N }, () => rng() < p);
        remaining.delete(node);
      } else {
        // Conditional
        const deps = parseDependencies(pDef);
        if ([...deps].every(d => d in states)) {
          states[node] = simulateSystem(pDef, states, N, rng);
          remaining.delete(node);
        }
      }
    }
  }

  const results = {};
  for (const n of Object.keys(nodes)) {
    const arr = states[n];
    const sum = arr.reduce((a, v) => a + (v ? 1 : 0), 0);

    const mean = sum / N;
    const variance = arr.reduce((acc, v) => {
        const x = v ? 1 : 0;
        return acc + (x - mean) ** 2;
    }, 0) / (N - 1);
    
    let std = Math.sqrt(variance);

    if(std < 0.01) {
        std = Math.sqrt(variance).toExponential();
    } else {
        std = +std.toFixed(4)
    }

    results[n] = { mean, std }
  }
  return results;
}

// // Example
// const nodes = {
//   r: { p: [91, 11] }, // mean ≈ 91/(91+11) = 0.892
//   q: { p: { "r1": [81,21], "r0": [1,10] } }
// };

// console.log(simulate({ nodes, N: 100000, seed: 42 }));
