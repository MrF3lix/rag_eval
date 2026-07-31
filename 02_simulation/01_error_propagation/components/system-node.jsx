import { useContext } from 'react'
import { Node } from './node'
import { GraphContext } from './graph-context'
import { DebouncedInput } from './form/debounced-input'


export const SystemNode = ({ id, data, ...rest }) => {
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

        {isNaN(probabilities) && Object.keys(probabilities).map(field => {
          return (
            <div key={field} className="flex gap-2">
              <DebouncedInput
                label={`${field}`}
                value={probabilities[field]}
                className="flex-1"
                callback={(value) => updateConditionalProbability(field, value)}
                validate={(value => !isNaN(value) && !isNaN(parseFloat(value)))}
              />
            </div>

          )
        })}
        {!isNaN(probabilities) && (
          <div key={id} className="flex gap-2">
            <DebouncedInput
              label={`${id}`}
              value={probabilities}
              className="flex-1"
              callback={(value) => updateSingleProbability(value)}
              validate={(value => !isNaN(value) && !isNaN(parseFloat(value)))}
            />
          </div>

        )}

        <p className='text-2xl'>
          P({id}=1) = {successRates[id]}
        </p>

      </div>
    </Node>
  )
}