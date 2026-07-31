export const toReactFlowEdge = (edge) => ({
    'id': edge.id,
    'source': edge.fromNode,
    'sourceHandle': edge.fromHandle,
    'target': edge.toNode,
    'targetHandle': edge.toHandle,
    'animated': 'true',
    'type': 'default'
})