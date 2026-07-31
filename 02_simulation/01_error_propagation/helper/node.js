export const updateNodeFields = (node, id, fields) => {
    if (node.id !== id) return node
    return {
        ...node,
        data: {
            ...node.data,
            fields
        }
    }
}