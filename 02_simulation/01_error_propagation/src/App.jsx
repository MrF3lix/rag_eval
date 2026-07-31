import { useState, useMemo } from 'react'
import '@xyflow/react/dist/base.css'
import { Button } from '../components/form/button'
import { Graph } from '../components/graph'
import { Input } from '../components/form/input'
import { computeResult } from '../helper/api'
import { GraphContext } from '../components/graph-context'
import { useDebouncedEffect, useThrottledEffect } from '@react-hookz/web'
import { Settings } from '../components/settings'
import { AdjustmentsHorizontalIcon, ArrowDownIcon } from '@heroicons/react/24/outline'

const initial_conditional_probs = {
  "q": {
    "p": 0.95
  },
  // "r": {
  //   "p": {
  //     "q1": 0.7,
  //     "q0": 0
  //   }
  // },
  // "g": {
  //   "p": {
  //     "r1_q1": 0.8,
  //     "r1_q0": 0.5,
  //     "r0_q1": 0.5,
  //     "r0_q0": 0
  //   }
  // }
}
const initial_conditional_uncertainties = {
  "q": {
    "p": [101, 1]
  },
  "r": {
    "p": {
      "q1": [701, 301],
      "q0": [1, 1]
    }
  }
  // "r": {
  //   "p": {
  //     "q1": 0.7,
  //     "q0": 0
  //   }
  // },
  // "g": {
  //   "p": {
  //     "r1_q1": 0.8,
  //     "r1_q0": 0.5,
  //     "r0_q1": 0.5,
  //     "r0_q0": 0
  //   }
  // }
}
const initial_success_rates = Object.keys(initial_conditional_uncertainties).reduce((a, v) => ({ ...a, [v]: 0.0 }), {})

console.log(initial_success_rates)

export default function App() {
  const [showSettings, setShowSettings] = useState(false)
  const [settings, setSettings]= useState({N: 10_000, seed: 0})
  const [successRates, setSuccessRates] = useState(initial_success_rates)
  const [conditionalProbabilities, setConditionalProbabilities] = useState(initial_conditional_uncertainties)
  const value = { conditionalProbabilities, setConditionalProbabilities, successRates, setSuccessRates }

  useDebouncedEffect(() => {
    computeResult(settings, conditionalProbabilities, setSuccessRates)
    // setSuccessRates(result)

  }, [conditionalProbabilities, settings], 400)

  return (
    <GraphContext value={value}>
      <div className='w-[100vw] min-h-[100vh]'>
        <div className='relative'>
          <div className='absolute w-full top-0 left-0 z-100'>
            {showSettings &&
              <Settings settings={settings} setSettings={setSettings}/>
            }
            <div className='w-full bg-yellow-50 p-2 flex items-center justify-center cursor-pointer shadow-xs' onClick={() => setShowSettings(!showSettings)}>
              <AdjustmentsHorizontalIcon className='w-4 h-4'/>
              {/* <Button primary small onClick={() => setShowSettings(!showSettings)}>{showSettings ? 'Hide': 'Show'} Settings</Button> */}
            </div>
          </div>
          <div className='w-full h-[100vh]'>
            <Graph />
          </div>
        </div>
      </div>

    </GraphContext>
  )
}