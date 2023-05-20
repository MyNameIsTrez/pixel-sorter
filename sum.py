import os
from pathlib import Path

import numpy as np
import pyopencl as cl

a_np = np.random.rand(50000).astype(np.float32)
b_np = np.random.rand(50000).astype(np.float32)

# This environment variable needs to be set
# to let pyopencl not ask which platform (GPU) to use
os.environ["PYOPENCL_CTX"] = "0"
ctx = cl.create_some_context()

queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

prg = cl.Program(ctx, Path("sum.cl").read_text()).build()

res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)

knl = prg.sum
knl(queue, a_np.shape, None, a_g, b_g, res_g)

res_np = np.empty_like(a_np)
cl.enqueue_copy(queue, res_np, res_g)

# Assert GPU result being identical to CPU numpy result
assert np.allclose(res_np, a_np + b_np)
