import { useContext, useState } from "react"
import { Input } from "./form/input"
import { Button } from "./form/button"
import { GraphContext } from "./graph-context";
import { DebouncedInput } from "./form/debounced-input";

export const Settings = ({ settings, setSettings }) => {
    const { conditionalProbabilities, setConditionalProbabilities } = useContext(GraphContext);

    const [newNode, setNewNode] = useState("")


    const addNode = () => {
        const key = newNode[0]

        setConditionalProbabilities({
            ...conditionalProbabilities,
            [key]: {
                p: 0.0
            }
        })
        setNewNode("")
    }

    const submit = (e) => {
        e.preventDefault()
        if(newNode.length == 0) {
            return
        }

        addNode()
    }

    return (
        <div className='flex flex-col gap-2 bg-amber-50 p-4'>
            <h1 className="text-xl">System Failure Simulator</h1>
            <p className="text-sm text-gray-500">Add / Remove Nodes to represent your conditional system and set the conditional probabilities.</p>

            <div className="flex flex-col sm:flex-row gap-4 w-full">
                <div className="flex-1">
                    <h2>Settings</h2>
                    <div className="flex flex-row gap-2 items-start">
                        <DebouncedInput
                            label="N"
                            placeholder="100"
                            value={settings.N}
                            validate={value => !isNaN(value)}
                            callback={value => setSettings({ ...settings, N: value })}
                            className="flex-1"
                        />
                        <DebouncedInput
                            label="Seed"
                            placeholder="seed"
                            value={settings.seed}
                            validate={value => !isNaN(value)}
                            callback={value => setSettings({ ...settings, seed: value })}
                            className="flex-1"
                        />
                    </div>
                </div>
                <div className="flex-1">
                    <h2>Add Node</h2>
                    <form onSubmit={submit} className="flex flex-col gap-2">
                        <DebouncedInput
                            delay={0}
                            label="New Node Name"
                            placeholder="Name"
                            value={newNode}
                            validate={value => value.length <= 1  ? true : 'Node name must be only one character'}
                            callback={value => setNewNode(value)}
                            className="flex-1"
                        />
                        <Button primary >
                            Add Node
                        </Button>
                    </form>

                </div>
            </div>
        </div>
    )
}