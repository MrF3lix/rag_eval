import { useContext } from 'react'
import { Node } from './node'
import { GraphContext } from './graph-context'
import { DebouncedInput } from './form/debounced-input'


export const SystemUncertaintyNode = ({ id, data, ...rest }) => {
  const { conditionalProbabilities, setConditionalProbabilities, successRates } = useContext(GraphContext)

  const updateConditionalProbability = (field, value) => {
    setConditionalProbabilities({
      ...conditionalProbabilities,
      [id]: {
        ...conditionalProbabilities[id],
        p: {
          ...conditionalProbabilities[id]['p'],
          [field]: parseFloat(value)
        }
      }
    })
  } 

  const updateSingleProbability = (value) => {
    setConditionalProbabilities({
      ...conditionalProbabilities,
      [id]: {
        p: parseFloat(value)
      }
    })
  }

  if (!conditionalProbabilities[id]) {
    return <></>
  }

  const probabilities = conditionalProbabilities[id]['p']
  return (
    <Node id={id} data={data} {...rest}>
      <div className="p-2 flex flex-col gap-3">

        {isNaN(probabilities[0]) && Object.keys(probabilities).map(field => {
          return (
            <div key={field} className="flex flex-col">
              <div>{field}</div>
              <div className='flex gap-2'>
                <DebouncedInput
                  label="Alpha"
                  value={probabilities[field][0]}
                  className="flex-1"
                  callback={(value) => updateConditionalProbability(field, value)}
                  validate={(value => !isNaN(value) && !isNaN(parseFloat(value)))}
                />
                <DebouncedInput
                  label="Beta"
                  value={probabilities[field][1]}
                  className="flex-1"
                  callback={(value) => updateConditionalProbability(field, value)}
                  validate={(value => !isNaN(value) && !isNaN(parseFloat(value)))}
                />
              </div>
            </div>

          )
        })}
        {!isNaN(probabilities[0]) && (
          <div key={id} className="flex flex-col">
              <div>{id}</div>
              <div className='flex gap-2'>
                <DebouncedInput
                  label="Alpha"
                  value={probabilities[0]}
                  className="flex-1"
                  callback={(value) => updateSingleProbability(value)}
                  validate={(value => !isNaN(value) && !isNaN(parseFloat(value)))}
                />
                <DebouncedInput
                  label="Beta"
                  value={probabilities[1]}
                  className="flex-1"
                  callback={(value) => updateSingleProbability(value)}
                  validate={(value => !isNaN(value) && !isNaN(parseFloat(value)))}
                />
              </div>
          </div>

        )}

        <p className='text-2xl'>
          P({id}=1) = {successRates[id]['mean']} ±{successRates[id]['std']}
        </p>

      </div>
    </Node>
  )
}