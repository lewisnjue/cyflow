Starting comprehensive tensor benchmarks
Benchmark shapes: 512x512, 1024x1024
Iterations per test: 5

Benchmarking shape 512x512...
Benchmarking shape 1024x1024...

============================================================================================================================================
Library    | Device | Shape      | Mode     | Operation                |    Time (ms)
--------------------------------------------------------------------------------------------------------------------------------------------
NumPy      | cpu    | 512x512    | magic    | add_scalar_magic         |        0.095
NumPy      | cpu    | 512x512    | magic    | sub_scalar_magic         |        0.089
NumPy      | cpu    | 512x512    | magic    | mul_scalar_magic         |        0.087
NumPy      | cpu    | 512x512    | magic    | div_scalar_magic         |        0.094
NumPy      | cpu    | 512x512    | magic    | add_tensor_magic         |        0.143
NumPy      | cpu    | 512x512    | magic    | sub_tensor_magic         |        0.144
NumPy      | cpu    | 512x512    | magic    | mul_tensor_magic         |        0.140
NumPy      | cpu    | 512x512    | magic    | div_tensor_magic         |        0.141
NumPy      | cpu    | 512x512    | inplace  | add_scalar_inplace       |        0.121
NumPy      | cpu    | 512x512    | inplace  | sub_scalar_inplace       |        0.112
NumPy      | cpu    | 512x512    | inplace  | mul_scalar_inplace       |        0.109
NumPy      | cpu    | 512x512    | inplace  | div_scalar_inplace       |        0.130
NumPy      | cpu    | 512x512    | inplace  | add_tensor_inplace       |        0.168
NumPy      | cpu    | 512x512    | inplace  | sub_tensor_inplace       |        0.163
NumPy      | cpu    | 512x512    | inplace  | mul_tensor_inplace       |        0.165
NumPy      | cpu    | 512x512    | inplace  | div_tensor_inplace       |        0.167
PyTorch    | cuda   | 512x512    | magic    | add_scalar_magic         |        3.215
PyTorch    | cuda   | 512x512    | magic    | sub_scalar_magic         |        0.023
PyTorch    | cuda   | 512x512    | magic    | mul_scalar_magic         |        1.773
PyTorch    | cuda   | 512x512    | magic    | div_scalar_magic         |        2.087
PyTorch    | cuda   | 512x512    | magic    | add_tensor_magic         |        0.021
PyTorch    | cuda   | 512x512    | magic    | sub_tensor_magic         |        0.013
PyTorch    | cuda   | 512x512    | magic    | mul_tensor_magic         |        0.019
PyTorch    | cuda   | 512x512    | magic    | div_tensor_magic         |        0.021
PyTorch    | cuda   | 512x512    | inplace  | add_scalar_inplace       |        0.057
PyTorch    | cuda   | 512x512    | inplace  | sub_scalar_inplace       |        0.033
PyTorch    | cuda   | 512x512    | inplace  | mul_scalar_inplace       |        0.033
PyTorch    | cuda   | 512x512    | inplace  | div_scalar_inplace       |        0.035
PyTorch    | cuda   | 512x512    | inplace  | add_tensor_inplace       |        0.027
PyTorch    | cuda   | 512x512    | inplace  | sub_tensor_inplace       |        0.024
PyTorch    | cuda   | 512x512    | inplace  | mul_tensor_inplace       |        0.027
PyTorch    | cuda   | 512x512    | inplace  | div_tensor_inplace       |        0.023
Cyflow     | cuda   | 512x512    | magic    | add_scalar_magic         |        0.106
Cyflow     | cuda   | 512x512    | magic    | sub_scalar_magic         |        0.019
Cyflow     | cuda   | 512x512    | magic    | mul_scalar_magic         |        0.019
Cyflow     | cuda   | 512x512    | magic    | div_scalar_magic         |        0.019
Cyflow     | cuda   | 512x512    | magic    | add_tensor_magic         |        0.019
Cyflow     | cuda   | 512x512    | magic    | sub_tensor_magic         |        0.019
Cyflow     | cuda   | 512x512    | magic    | mul_tensor_magic         |        0.019
Cyflow     | cuda   | 512x512    | magic    | div_tensor_magic         |        0.020
Cyflow     | cuda   | 512x512    | inplace  | add_scalar_inplace       |        0.097
Cyflow     | cuda   | 512x512    | inplace  | sub_scalar_inplace       |        0.021
Cyflow     | cuda   | 512x512    | inplace  | mul_scalar_inplace       |        0.023
Cyflow     | cuda   | 512x512    | inplace  | div_scalar_inplace       |        0.020
Cyflow     | cuda   | 512x512    | inplace  | add_tensor_inplace       |        0.023
Cyflow     | cuda   | 512x512    | inplace  | sub_tensor_inplace       |        0.034
Cyflow     | cuda   | 512x512    | inplace  | mul_tensor_inplace       |        0.029
Cyflow     | cuda   | 512x512    | inplace  | div_tensor_inplace       |        0.025
NumPy      | cpu    | 1024x1024  | magic    | add_scalar_magic         |        0.410
NumPy      | cpu    | 1024x1024  | magic    | sub_scalar_magic         |        0.342
NumPy      | cpu    | 1024x1024  | magic    | mul_scalar_magic         |        0.337
NumPy      | cpu    | 1024x1024  | magic    | div_scalar_magic         |        0.343
NumPy      | cpu    | 1024x1024  | magic    | add_tensor_magic         |        0.551
NumPy      | cpu    | 1024x1024  | magic    | sub_tensor_magic         |        0.533
NumPy      | cpu    | 1024x1024  | magic    | mul_tensor_magic         |        0.537
NumPy      | cpu    | 1024x1024  | magic    | div_tensor_magic         |        0.540
NumPy      | cpu    | 1024x1024  | inplace  | add_scalar_inplace       |        0.512
NumPy      | cpu    | 1024x1024  | inplace  | sub_scalar_inplace       |        0.512
NumPy      | cpu    | 1024x1024  | inplace  | mul_scalar_inplace       |        0.507
NumPy      | cpu    | 1024x1024  | inplace  | div_scalar_inplace       |        0.558
NumPy      | cpu    | 1024x1024  | inplace  | add_tensor_inplace       |        0.686
NumPy      | cpu    | 1024x1024  | inplace  | sub_tensor_inplace       |        0.674
NumPy      | cpu    | 1024x1024  | inplace  | mul_tensor_inplace       |        0.680
NumPy      | cpu    | 1024x1024  | inplace  | div_tensor_inplace       |        0.682
PyTorch    | cuda   | 1024x1024  | magic    | add_scalar_magic         |        0.019
PyTorch    | cuda   | 1024x1024  | magic    | sub_scalar_magic         |        0.012
PyTorch    | cuda   | 1024x1024  | magic    | mul_scalar_magic         |        0.012
PyTorch    | cuda   | 1024x1024  | magic    | div_scalar_magic         |        0.013
PyTorch    | cuda   | 1024x1024  | magic    | add_tensor_magic         |        0.009
PyTorch    | cuda   | 1024x1024  | magic    | sub_tensor_magic         |        0.009
PyTorch    | cuda   | 1024x1024  | magic    | mul_tensor_magic         |        0.009
PyTorch    | cuda   | 1024x1024  | magic    | div_tensor_magic         |        0.009
PyTorch    | cuda   | 1024x1024  | inplace  | add_scalar_inplace       |        0.026
PyTorch    | cuda   | 1024x1024  | inplace  | sub_scalar_inplace       |        0.021
PyTorch    | cuda   | 1024x1024  | inplace  | mul_scalar_inplace       |        0.027
PyTorch    | cuda   | 1024x1024  | inplace  | div_scalar_inplace       |        0.021
PyTorch    | cuda   | 1024x1024  | inplace  | add_tensor_inplace       |        0.018
PyTorch    | cuda   | 1024x1024  | inplace  | sub_tensor_inplace       |        0.018
PyTorch    | cuda   | 1024x1024  | inplace  | mul_tensor_inplace       |        0.018
PyTorch    | cuda   | 1024x1024  | inplace  | div_tensor_inplace       |        0.017
Cyflow     | cuda   | 1024x1024  | magic    | add_scalar_magic         |        0.871
Cyflow     | cuda   | 1024x1024  | magic    | sub_scalar_magic         |        0.164
Cyflow     | cuda   | 1024x1024  | magic    | mul_scalar_magic         |        0.162
Cyflow     | cuda   | 1024x1024  | magic    | div_scalar_magic         |        0.167
Cyflow     | cuda   | 1024x1024  | magic    | add_tensor_magic         |        0.183
Cyflow     | cuda   | 1024x1024  | magic    | sub_tensor_magic         |        0.184
Cyflow     | cuda   | 1024x1024  | magic    | mul_tensor_magic         |        0.179
Cyflow     | cuda   | 1024x1024  | magic    | div_tensor_magic         |        0.181
Cyflow     | cuda   | 1024x1024  | inplace  | add_scalar_inplace       |        0.201
Cyflow     | cuda   | 1024x1024  | inplace  | sub_scalar_inplace       |        0.203
Cyflow     | cuda   | 1024x1024  | inplace  | mul_scalar_inplace       |        0.198
Cyflow     | cuda   | 1024x1024  | inplace  | div_scalar_inplace       |        0.199
Cyflow     | cuda   | 1024x1024  | inplace  | add_tensor_inplace       |        0.219
Cyflow     | cuda   | 1024x1024  | inplace  | sub_tensor_inplace       |        0.221
Cyflow     | cuda   | 1024x1024  | inplace  | mul_tensor_inplace       |        0.222
Cyflow     | cuda   | 1024x1024  | inplace  | div_tensor_inplace       |        0.216
============================================================================================================================================
