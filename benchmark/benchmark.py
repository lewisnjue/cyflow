import time
import numpy as np
import torch
import cyflow

def benchmark():
    # 2048 x 2048 = 4,194,304 elements (~16.7 MB of float32 data)
    shape = (2048, 2048)
    numel = np.prod(shape)
    iterations = 20  # Number of loops to average out timing fluctuations

    results = []

    print(f"Starting Benchmarks with Tensor Shape: {shape} ({numel:,} elements)")
    print(f"Averaging over {iterations} iterations per test...\n")

    # ==========================================
    # 1. NUMPY (CPU)
    # ==========================================
    print("Running NumPy (CPU)...")
    
    # Create & Fill Uniform
    start = time.perf_counter()
    for _ in range(iterations):
        np_arr = np.random.uniform(0, 1, shape).astype(np.float32)
    np_create_time = (time.perf_counter() - start) / iterations * 1000

    # Inplace Scalar Add
    start = time.perf_counter()
    for _ in range(iterations):
        np_arr += 1.0
    np_scalar_add = (time.perf_counter() - start) / iterations * 1000

    # Inplace Tensor Add
    np_arr2 = np.ones(shape, dtype=np.float32)
    start = time.perf_counter()
    for _ in range(iterations):
        np_arr += np_arr2
    np_tensor_add = (time.perf_counter() - start) / iterations * 1000

    results.append(("NumPy", "CPU", np_create_time, np_scalar_add, np_tensor_add))


    # ==========================================
    # 2. PYTORCH (CPU)
    # ==========================================
    print("Running PyTorch (CPU)...")
    
    start = time.perf_counter()
    for _ in range(iterations):
        pt_cpu = torch.rand(shape, dtype=torch.float32, device="cpu")
    pt_cpu_create = (time.perf_counter() - start) / iterations * 1000

    start = time.perf_counter()
    for _ in range(iterations):
        pt_cpu.add_(1.0)
    pt_cpu_scalar = (time.perf_counter() - start) / iterations * 1000

    pt_cpu2 = torch.ones(shape, dtype=torch.float32, device="cpu")
    start = time.perf_counter()
    for _ in range(iterations):
        pt_cpu.add_(pt_cpu2)
    pt_cpu_tensor = (time.perf_counter() - start) / iterations * 1000

    results.append(("PyTorch", "CPU", pt_cpu_create, pt_cpu_scalar, pt_cpu_tensor))


    # ==========================================
    # 3. PYTORCH (CUDA)
    # ==========================================
    print("Running PyTorch (CUDA)...")
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        pt_gpu = torch.rand(shape, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
    pt_gpu_create = (time.perf_counter() - start) / iterations * 1000

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        pt_gpu.add_(1.0)
        torch.cuda.synchronize()
    pt_gpu_scalar = (time.perf_counter() - start) / iterations * 1000

    pt_gpu2 = torch.ones(shape, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        pt_gpu.add_(pt_gpu2)
        torch.cuda.synchronize()
    pt_gpu_tensor = (time.perf_counter() - start) / iterations * 1000

    results.append(("PyTorch", "CUDA", pt_gpu_create, pt_gpu_scalar, pt_gpu_tensor))


    # ==========================================
    # 4. CYFLOW (CPU)
    # ==========================================
    print("Running Cyflow (CPU)...")
    cyflow.manual_seed(42, device=cyflow.CPU)
    
    start = time.perf_counter()
    for _ in range(iterations):
        cf_cpu = cyflow.tensor(shape=shape, device=cyflow.CPU)
        cf_cpu.fill_uniform()
    cf_cpu_create = (time.perf_counter() - start) / iterations * 1000

    start = time.perf_counter()
    for _ in range(iterations):
        cf_cpu += 1.0
    cf_cpu_scalar = (time.perf_counter() - start) / iterations * 1000

    cf_cpu2 = cyflow.tensor(shape=shape, device=cyflow.CPU)
    cf_cpu2[:] = 1.0
    start = time.perf_counter()
    for _ in range(iterations):
        cf_cpu += cf_cpu2
    cf_cpu_tensor = (time.perf_counter() - start) / iterations * 1000

    results.append(("Cyflow", "CPU", cf_cpu_create, cf_cpu_scalar, cf_cpu_tensor))


    # ==========================================
    # 5. CYFLOW (CUDA)
    # ==========================================
    print("Running Cyflow (CUDA)...")
    cyflow.manual_seed(42, device=cyflow.CUDA)
    
    start = time.perf_counter()
    for _ in range(iterations):
        cf_gpu = cyflow.tensor(shape=shape, device=cyflow.CUDA)
        cf_gpu.fill_uniform()
    cf_gpu_create = (time.perf_counter() - start) / iterations * 1000

    start = time.perf_counter()
    for _ in range(iterations):
        cf_gpu += 1.0
    cf_gpu_scalar = (time.perf_counter() - start) / iterations * 1000

    cf_gpu2 = cyflow.tensor(shape=shape, device=cyflow.CUDA)
    cf_gpu2[:] = 1.0
    start = time.perf_counter()
    for _ in range(iterations):
        cf_gpu += cf_gpu2
    cf_gpu_tensor = (time.perf_counter() - start) / iterations * 1000

    results.append(("Cyflow", "CUDA", cf_gpu_create, cf_gpu_scalar, cf_gpu_tensor))


    # ==========================================
    # FORMATTED RESULTS TABLE
    # ==========================================
    print("\n" + "=" * 90)
    print(f"{'Framework':<10} | {'Device':<6} | {'Create + Fill (ms)':<20} | {'Scalar Add (ms)':<18} | {'Tensor Add (ms)':<18}")
    print("-" * 90)
    for row in results:
        print(f"{row[0]:<10} | {row[1]:<6} | {row[2]:<20.4f} | {row[3]:<18.4f} | {row[4]:<18.4f}")
    print("=" * 90)

if __name__ == "__main__":
    benchmark()