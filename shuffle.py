import os
from pathlib import Path

import numpy as np
import pyopencl as cl

arr_np = np.random.rand(5).astype(np.float32)

# This environment variable needs to be set
# to let pyopencl not ask which platform (GPU) to use
os.environ["PYOPENCL_CTX"] = "0"
ctx = cl.create_some_context()

queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
arr = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=arr_np)

prg = cl.Program(ctx, Path("shuffle.cl").read_text()).build()

knl = prg.shuffle_
knl(queue, arr_np.shape, None, arr)

res_np = np.empty_like(arr_np)
cl.enqueue_copy(queue, res_np, arr)

print(arr_np)
print(res_np)
