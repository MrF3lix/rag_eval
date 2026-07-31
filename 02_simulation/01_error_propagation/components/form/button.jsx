import { classNames } from '../../helper/class'

export const Button = ({ onClick, disabled = false, primary = false, secondary = false, small = false, dangerous = false, children, ...rest }) => (
    <button
        className={classNames(
            'rounded-md',
            primary ? 'bg-blue-500 hover:bg-blue-400 text-gray-100' : '',
            secondary ? 'bg-gray-300 hover:bg-gray-200 text-gray-800 dark:bg-gray-300' : '',
            dangerous ? 'bg-red-500 text-gray-100 disabled:bg-red-200 disabled:text-gray-50 dark:disabled:bg-red-400 dark:disabled:opacity-60' : 'disabled:bg-gray-100 dark:disabled:bg-gray-700 disabled:text-gray-300 dark:disabled:text-gray-500 ',
            small ? 'px-4 py-2 text-sm' : 'px-6 py-2'
        )}
        onClick={onClick}
        disabled={disabled}
        {...rest}
    >
        {children}
    </button>
)
