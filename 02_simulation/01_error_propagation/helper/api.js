import SimulationWorker from '../worker/simulate_uncertainty?worker'

export const computeResult = async (settings, nodes, setSuccessRates) => {
    try {
        const worker = new SimulationWorker();
        console.time('Calc')
        worker.postMessage({
                nodes: nodes,
                ...settings
        })

        worker.onmessage = (e) => {
            console.timeEnd('Calc')
            setSuccessRates(e.data)
        }
    } catch (error) {
        console.log(error)
        return error
    }
}