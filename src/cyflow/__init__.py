from .tensor import Tensor, CPU, CUDA, manual_seed

USE_CUDA = False
def _infer_shape_and_flatten(data):
    """
    Recursively inspects a nested list to determine its shape
    and flattens the elements into a single 1D list.
    """
    if isinstance(data, (int, float)):
        return [], [float(data)]

    if isinstance(data, list):
        if len(data) == 0:
            return [0], []

        inner_shapes = []
        flat_data = []

        for item in data:
            shape, flat = _infer_shape_and_flatten(item)
            inner_shapes.append(shape)
            flat_data.extend(flat)

        first_shape = inner_shapes[0]
        for s in inner_shapes[1:]:
            if s != first_shape:
                raise ValueError(
                    f"Inconsistent tensor dimensions: expected {first_shape}, got {s}"
                )

        return [len(data)] + first_shape, flat_data

    raise TypeError(f"Unsupported data type for tensor: {type(data)}")


def tensor(data=None, shape: tuple = None, device: int = CPU) -> Tensor:
    """
    Creates a cyflow Tensor.
    Can be initialized from data, an empty shape, or both.
    """
    if data is None and shape is None:
        raise ValueError(
            "You must provide either 'data', 'shape', or both to create a tensor."
        )

    if data is None:
        if isinstance(shape, int):
            shape = [shape]
        t = Tensor(shape, device=device)
        t.fill_uniform()
        return t

    inferred_shape, flat_data = _infer_shape_and_flatten(data)

    final_shape = shape if shape is not None else inferred_shape

    expected_numel = 1
    for dim in final_shape:
        expected_numel *= dim

    if len(flat_data) != expected_numel:
        raise ValueError(
            f"Shape {final_shape} is invalid for input with {len(flat_data)} elements"
        )

    t = Tensor(final_shape, device=device)

    t._set_data_from_list(flat_data)

    return t
