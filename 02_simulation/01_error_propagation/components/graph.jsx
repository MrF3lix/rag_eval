import { useCallback, useContext, useEffect, useMemo } from 'react'
import { ReactFlow, ReactFlowProvider, Controls, Background, BackgroundVariant, MiniMap, useNodesState, useEdgesState } from '@xyflow/react'
import { SystemNode } from '../components/system-node'
import dagre from '@dagrejs/dagre'

import '@xyflow/react/dist/base.css'
import { addEdgeToDependencyGraph, computeFieldCombinations, graphToEdges, graphToNodes, removeEdgeFromDependencyGraph } from '../helper/dependency'
import { GraphContext, inferDependencyGraph } from './graph-context'
import { useDeepCompareMemo } from '@react-hookz/web'
import { SystemUncertaintyNode } from './system-uncertainty-node'

const nodeTypes = {
    system: SystemNode,
    systemUncertainty: SystemUncertaintyNode
}

const nodeWidth = 500
const nodeHeight = 400


const dagreGraph = new dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}))

const getLayoutedElements = (nodes, edges, direction = 'LR') => {
    const isHorizontal = direction === 'LR'
    dagreGraph.setGraph({ rankdir: direction })

    nodes.forEach((node) => {
        dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
    })

    edges.forEach((edge) => {
        dagreGraph.setEdge(edge.source, edge.target)
    })

    dagre.layout(dagreGraph)

    const newNodes = nodes.map((node) => {
        const nodeWithPosition = dagreGraph.node(node.id);
        const newNode = {
            ...node,
            targetPosition: isHorizontal ? 'left' : 'top',
            sourcePosition: isHorizontal ? 'right' : 'bottom',
            position: {
                x: nodeWithPosition.x - nodeWidth / 2,
                y: nodeWithPosition.y - nodeHeight / 2,
            },
        }

        return newNode
    })

    return { nodes: newNodes, edges }
}

export const Graph = ({ nodeType = 'systemUncertainty' }) => {
    const { conditionalProbabilities, setConditionalProbabilities } = useContext(GraphContext);

    const [nodes, setNodes, onNodesChange] = useNodesState();
    const [edges, setEdges, onEdgesChange] = useEdgesState();

    useDeepCompareMemo(() => {
        const dependencyGraph = inferDependencyGraph(conditionalProbabilities)
        const initialNodes = graphToNodes(dependencyGraph, nodeType)
        const initialEdges = graphToEdges(dependencyGraph)
        const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(initialNodes, initialEdges,)

        setNodes(layoutedNodes)
        setEdges(layoutedEdges)

    }, [conditionalProbabilities])

    const onConnect = useCallback((newEdge) => {
        const dependencyGraph = inferDependencyGraph(conditionalProbabilities)
        const newDependencyGraph = addEdgeToDependencyGraph(newEdge, dependencyGraph)
        const combinations = computeFieldCombinations(newDependencyGraph, newEdge)

        setConditionalProbabilities({
            ...conditionalProbabilities,
            [newEdge.target]: {
                p: combinations
            }
        })

        setEdges([...edges, newEdge])
    })

    const onEdgesDelete = useCallback((removedEdges) => {
        const dependencyGraph = inferDependencyGraph(conditionalProbabilities)
        const newDependencyGraph = removeEdgeFromDependencyGraph(removedEdges[0], dependencyGraph)
        const combinations = computeFieldCombinations(newDependencyGraph, removedEdges[0])

        setConditionalProbabilities({
            ...conditionalProbabilities,
            [removedEdges[0].target]: {
                p: combinations
            }
        })
        setEdges(edges.filter(e => e.id != removedEdges[0].id))
    })

    return (
        <ReactFlowProvider>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                snapToGrid={true}
                snapGrid={[15, 15]}
                nodeTypes={nodeTypes}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onEdgesDelete={onEdgesDelete}
                onConnect={onConnect}
                fitView
            >
                <Controls />
                <MiniMap />
                <Background variant={BackgroundVariant.Dots} />
            </ReactFlow>
        </ReactFlowProvider>
    )
}