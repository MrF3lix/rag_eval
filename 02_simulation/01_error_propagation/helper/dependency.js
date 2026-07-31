const f = (a, b) => [].concat(...a.map(d => b.map(e => [].concat(d, e))));
const cartesian = (a, b, ...c) => (b ? cartesian(f(a, b), ...c) : a);

export const computeFieldCombinations = (dependencies, newEdge) => {
  const target = newEdge.target
  const dep = dependencies[target]

  const keys = dep.map(d => ([`${d}0`, `${d}1`]))
  let pairs = cartesian(...keys)

  if (pairs == undefined) {
    return 0
  }

  if (Array.isArray(pairs[0])) {
    pairs = pairs.map(p => p.join('_'))
  }

  return pairs.reduce((a, v) => ({ ...a, [v]: 0.0 }), {})

}

export const removeNodeFromDependencyGraph = (id, dependencyGraph) => {
  let newD = {
    ...dependencyGraph
  }

  delete newD[id]

  Object.keys(newD).forEach(k => {
    newD[k] = newD[k].filter(r => r != id)
  })

  return newD
}

export const removeEdgeFromDependencyGraph = (params, dependencyGraph) => {
  return {
    ...dependencyGraph,
    [params.target]: [...dependencyGraph[params.target].filter(i => i != params.source)]
  }
}

export const addEdgeToDependencyGraph = (newEdge, dependencyGraph) => {
  return {
    ...dependencyGraph,
    [newEdge.target]: [
      ...dependencyGraph[newEdge.target],
      newEdge.source
    ]
  }
}

// export const updateNodesFromDependencyGraph = (dependencyGraph, nodes, updatedEdge) => {
//   const updatedNodes = graphToNodes(dependencyGraph)
//   const prevTargetNode = nodes.find(n => n.id == updatedEdge.target)
//   const targetNode = updatedNodes.find(n => n.id == updatedEdge.target)

//   const newNodes = [
//     ...nodes.filter(n => n.id != updatedEdge.target),
//     {
//       ...targetNode,
//       position: prevTargetNode.position
//     }
//   ]

//   return newNodes
// }

export const edgesToGraph = (edges) => {
  const graph = {}

  for (const { source, target } of edges) {
    if (!graph[target]) graph[target] = []
    graph[target].push(source)

    if (!graph[source]) graph[source] = []
  }

  return graph
}

export const graphToNodes = (graph, nodeType = 'system') => {
  const nodes = Object.keys(graph).map((k) => {
    return {
      'id': `${k}`,
      'type': nodeType,
      'data': {
        'title': k,
        'incomingHandles': [
          {
            "id": `${k}_in`,
            "name": `${k}_in`
          }
        ],
        'outgoingHandles': [
          {
            "id": `${k}_out`,
            "name": `${k}_out`
          }
        ]
      },
      'position': { x: 0, y: 0 }
    }
  })
  return nodes
}

export const graphToEdges = graph => {

  let edges = []

  Object.keys(graph).forEach((k, i) => {
    const dependsOn = graph[k]

    const e = dependsOn.map((dk, n) => ({
      "id": `${i}_${n}`,
      "source": `${dk}`,
      "sourceHandle": `${dk}_out`,
      "target": `${k}`,
      "targetHandle": `${k}_in`
    }))

    edges = [...edges, ...e]
  })

  return edges
}
