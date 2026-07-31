import { PlusIcon } from '@heroicons/react/24/outline'
import { XMarkIcon } from '@heroicons/react/24/outline'
import { EllipsisVerticalIcon } from '@heroicons/react/24/outline'
import { Handle, Position } from "@xyflow/react"
import { useContext } from 'react'
import { GraphContext, inferDependencyGraph } from './graph-context'
import { computeFieldCombinations, removeNodeFromDependencyGraph } from '../helper/dependency'


export const Node = ({ id, data, children }) => {
    const { conditionalProbabilities, setConditionalProbabilities } = useContext(GraphContext);

    const deleteNode = async () => {

        const dependencyGraph = inferDependencyGraph(conditionalProbabilities)
        const newDependencyGraph = removeNodeFromDependencyGraph(id, dependencyGraph)

        const newConditionalProbabilities = {}
        Object.keys(newDependencyGraph).map(k => {
            newConditionalProbabilities[k] = {
                p: computeFieldCombinations(newDependencyGraph, {target: k})
            }
        })

        setConditionalProbabilities(newConditionalProbabilities)
    }

    return (
        <>
            <div className="flex bg-gray-100 dark:bg-gray-900">
                {data.incomingHandles &&
                    <div className="w-0 flex flex-col justify-evenly items-center gap-1 relative">
                        {data.incomingHandles.map((input) => (
                            <div key={input.id}>
                                <Handle
                                    key={input.id}
                                    id={input.id}
                                    type="target"
                                    position={Position.Left}
                                    className="bg-yellow-400 react-flow__handle--input"
                                />
                            </div>
                        ))}
                        {data.canAddTargetHandle &&
                            <div
                                onClick={data.onAddedTargetHandle}
                                className=" bg-yellow-400 cursor-pointer pointer-events-auto flex justify-center items-center react-flow__add__handle"
                            >
                                <PlusIcon className="w-3 h-3" />
                            </div>
                        }
                    </div>
                }
                <div className="flex flex-col flex-1 text-gray-900 dark:text-gray-100 w-80">
                    <div className="p-2 flex justify-between items-center border-b border-gray-200 dark:border-gray-100">
                        <span className="custom-drag-handle">
                            <EllipsisVerticalIcon className="w-4 h-4" />
                        </span>
                        <span className="custom-drag-handle">
                            {data.title}
                        </span>
                        <div className="flex align-center">
                            <button onClick={deleteNode} className="custom-delete-handle w-4 cursor-pointer">
                                <XMarkIcon className="w-4 h-4" />
                            </button>
                        </div>
                    </div>
                    <div>
                        {children}
                    </div>
                </div>
                {data.outgoingHandles &&
                    <div className="w-2 flex flex-col justify-between gap-1">
                        {data.outgoingHandles.map((output) => (
                            <Handle
                                key={output.id}
                                id={output.id}
                                type="source"
                                position={Position.Right}
                                className="flex-1 grow bg-yellow-400 react-flow__handle--out"
                            />
                        ))}
                    </div>
                }
            </div>
        </>
    )
}