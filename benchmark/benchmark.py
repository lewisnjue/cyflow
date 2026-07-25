import time
import numpy as np
import torch
import cyflow


def _time_ms(fn, iterations):
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) / iterations * 1000.0


def _format_shape(shape):
    return f"{shape[0]}x{shape[1]}"


def benchmark_numpy(shape, iterations=5):
    np.random.seed(0)
    base = np.random.uniform(0.0, 1.0, size=shape).astype(np.float32)
    other = np.random.uniform(0.0, 1.0, size=shape).astype(np.float32)
    rows = []

    operations = [
        ("add_scalar_magic", lambda: base + 1.0),
        ("sub_scalar_magic", lambda: base - 1.0),
        ("mul_scalar_magic", lambda: base * 1.5),
        ("div_scalar_magic", lambda: base / 2.0),
        ("add_tensor_magic", lambda: base + other),
        ("sub_tensor_magic", lambda: base - other),
        ("mul_tensor_magic", lambda: base * other),
        ("div_tensor_magic", lambda: base / other),
    ]

    for name, fn in operations:
        rows.append(("NumPy", "cpu", _format_shape(shape), "magic", name, _time_ms(fn, iterations)))

    inplace_ops = [
        ("add_scalar_inplace", lambda: (lambda a: (a.__iadd__(1.0), a)[1])(base.copy())),
        ("sub_scalar_inplace", lambda: (lambda a: (a.__isub__(1.0), a)[1])(base.copy())),
        ("mul_scalar_inplace", lambda: (lambda a: (a.__imul__(1.5), a)[1])(base.copy())),
        ("div_scalar_inplace", lambda: (lambda a: (a.__itruediv__(2.0), a)[1])(base.copy())),
        ("add_tensor_inplace", lambda: (lambda a: (a.__iadd__(other), a)[1])(base.copy())),
        ("sub_tensor_inplace", lambda: (lambda a: (a.__isub__(other), a)[1])(base.copy())),
        ("mul_tensor_inplace", lambda: (lambda a: (a.__imul__(other), a)[1])(base.copy())),
        ("div_tensor_inplace", lambda: (lambda a: (a.__itruediv__(other), a)[1])(base.copy())),
    ]

    for name, fn in inplace_ops:
        rows.append(("NumPy", "cpu", _format_shape(shape), "inplace", name, _time_ms(fn, iterations)))

    return rows


def benchmark_torch(shape, iterations=5):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the PyTorch benchmark")

    torch.manual_seed(0)
    base = torch.rand(shape, dtype=torch.float32, device="cuda")
    other = torch.rand(shape, dtype=torch.float32, device="cuda")
    rows = []

    operations = [
        ("add_scalar_magic", lambda: base + 1.0),
        ("sub_scalar_magic", lambda: base - 1.0),
        ("mul_scalar_magic", lambda: base * 1.5),
        ("div_scalar_magic", lambda: base / 2.0),
        ("add_tensor_magic", lambda: base + other),
        ("sub_tensor_magic", lambda: base - other),
        ("mul_tensor_magic", lambda: base * other),
        ("div_tensor_magic", lambda: base / other),
    ]

    for name, fn in operations:
        rows.append(("PyTorch", "cuda", _format_shape(shape), "magic", name, _time_ms(fn, iterations)))

    inplace_ops = [
        ("add_scalar_inplace", lambda: (lambda a: (a.__iadd__(1.0), a)[1])(base.clone())),
        ("sub_scalar_inplace", lambda: (lambda a: (a.__isub__(1.0), a)[1])(base.clone())),
        ("mul_scalar_inplace", lambda: (lambda a: (a.__imul__(1.5), a)[1])(base.clone())),
        ("div_scalar_inplace", lambda: (lambda a: (a.__itruediv__(2.0), a)[1])(base.clone())),
        ("add_tensor_inplace", lambda: (lambda a: (a.__iadd__(other), a)[1])(base.clone())),
        ("sub_tensor_inplace", lambda: (lambda a: (a.__isub__(other), a)[1])(base.clone())),
        ("mul_tensor_inplace", lambda: (lambda a: (a.__imul__(other), a)[1])(base.clone())),
        ("div_tensor_inplace", lambda: (lambda a: (a.__itruediv__(other), a)[1])(base.clone())),
    ]

    for name, fn in inplace_ops:
        rows.append(("PyTorch", "cuda", _format_shape(shape), "inplace", name, _time_ms(fn, iterations)))

    return rows


def benchmark_cyflow(shape, iterations=5):
    cyflow.manual_seed(42, device=cyflow.CUDA)
    base = cyflow.tensor(shape=shape, device=cyflow.CUDA)
    other = cyflow.tensor(shape=shape, device=cyflow.CUDA)
    rows = []

    operations = [
        ("add_scalar_magic", lambda: base + 1.0),
        ("sub_scalar_magic", lambda: base - 1.0),
        ("mul_scalar_magic", lambda: base * 1.5),
        ("div_scalar_magic", lambda: base / 2.0),
        ("add_tensor_magic", lambda: base + other),
        ("sub_tensor_magic", lambda: base - other),
        ("mul_tensor_magic", lambda: base * other),
        ("div_tensor_magic", lambda: base / other),
    ]

    for name, fn in operations:
        rows.append(("Cyflow", "cuda", _format_shape(shape), "magic", name, _time_ms(fn, iterations)))

    inplace_ops = [
        ("add_scalar_inplace", lambda: (lambda a: (a.__iadd__(1.0), a)[1])(base.detach())),
        ("sub_scalar_inplace", lambda: (lambda a: (a.__isub__(1.0), a)[1])(base.detach())),
        ("mul_scalar_inplace", lambda: (lambda a: (a.__imul__(1.5), a)[1])(base.detach())),
        ("div_scalar_inplace", lambda: (lambda a: (a.__itruediv__(2.0), a)[1])(base.detach())),
        ("add_tensor_inplace", lambda: (lambda a: (a.__iadd__(other), a)[1])(base.detach())),
        ("sub_tensor_inplace", lambda: (lambda a: (a.__isub__(other), a)[1])(base.detach())),
        ("mul_tensor_inplace", lambda: (lambda a: (a.__imul__(other), a)[1])(base.detach())),
        ("div_tensor_inplace", lambda: (lambda a: (a.__itruediv__(other), a)[1])(base.detach())),
    ]

    for name, fn in inplace_ops:
        rows.append(("Cyflow", "cuda", _format_shape(shape), "inplace", name, _time_ms(fn, iterations)))

    return rows


def benchmark():
    shapes = [(512, 512), (1024, 1024)]
    iterations = 5

    print("Starting comprehensive tensor benchmarks")
    print(f"Benchmark shapes: {', '.join(_format_shape(shape) for shape in shapes)}")
    print(f"Iterations per test: {iterations}\n")

    all_rows = []
    for shape in shapes:
        print(f"Benchmarking shape { _format_shape(shape) }...")
        all_rows.extend(benchmark_numpy(shape, iterations=iterations))
        all_rows.extend(benchmark_torch(shape, iterations=iterations))
        all_rows.extend(benchmark_cyflow(shape, iterations=iterations))

    print("\n" + "=" * 140)
    print(f"{'Library':<10} | {'Device':<6} | {'Shape':<10} | {'Mode':<8} | {'Operation':<24} | {'Time (ms)':>12}")
    print("-" * 140)
    for row in all_rows:
        print(f"{row[0]:<10} | {row[1]:<6} | {row[2]:<10} | {row[3]:<8} | {row[4]:<24} | {row[5]:>12.3f}")
    print("=" * 140)


if __name__ == "__main__":
    benchmark()