import { classNames } from "../../helper/class";

export const Input = ({ label, value, defaultValue, onChange, className, isValid = true, disabled = false, error = null, ...rest }) => (
    <label className={`text-left ${className}`}>
        <span className="text-gray-600 dark:text-gray-300 text-xs">{label}</span>
        <input
            value={value}
            onChange={onChange}
            disabled={disabled}
            className={
                classNames(
                    `p-2 border rounded-sm w-full bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 text-sm  disabled:text-gray-400 disabled:bg-gray-200 disabled:border-gray-200 disabled:dark:text-gray-500 disabled:dark:bg-gray-700 disabled:dark:border-gray-700`,
                    !isValid ? 'border-red-500 focus-within:outline-red-500' : ''
                )}
            defaultValue={defaultValue}
            {...rest}
        />
        {error &&
            <span className="text-sm text-red-500">{error}</span>
        }
    </label>
)