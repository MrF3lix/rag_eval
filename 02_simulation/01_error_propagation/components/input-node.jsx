import { useCallback } from 'react'
import { Node } from './node'
import { useReactFlow } from '@xyflow/react'
import { Input } from './form/input'
import { Select } from './form/select'
import { updateNodeFields } from '../helper/node'

const httpMethods = [
  { value: 'get', name: 'GET' },
  { value: 'post', name: 'POST' },
  { value: 'put', name: 'PUT' },
  { value: 'patch', name: 'PATCH' },
  { value: 'delete', name: 'DELETE' },
]

export const InputNode = ( { id, data, ...rest } ) => {
  const reactFlowHook = useReactFlow()
  const updateNode = useCallback( ( params ) => {
    reactFlowHook.setNodes( reactFlowHook.getNodes().map( n => updateNodeFields( n, id, params ) ) )
  }, [id, reactFlowHook] )

  return (
    <Node id={id} data={data} {...rest}>
      <div className="p-2 flex flex-col gap-3">
        <Input
          label="Endpoint"
          value={data.label}
          disabled={true}
        />
        <Select
          label="HTTP Method"
          options={httpMethods}
          selected={false}
          onChange={( e ) => updateNode( { ...data.params, method: e.target.value } )}
        />
      </div>
    </Node>
  )
}