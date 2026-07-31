import { useState } from "react"
import { Input } from "./input"
import { useDebouncedEffect } from "@react-hookz/web"

export const DebouncedInput = ({callback, delay=100, validate=() => true, ...params}) => {
    const [value, setValue] = useState(params.value)
    const [isValid, setIsValid] = useState(true)

    useDebouncedEffect(() => {
        if(validate(value) === true) {
            setIsValid(true)
            callback(value)
        } else {
            setIsValid(false)
            return
        }
    }, [value], delay)

    return (
        <Input
            {...params}
            isValid={isValid}
            value={value}
            error={validate(value)}
            onChange={(e) => setValue(e.target.value)}
        />
    )
}