import ctypes
import torch
import ctypes as ct
import os
import numpy as np

dtype = torch.double
inp = torch.tensor([1.0,0.1], dtype=dtype)

inp1 = inp[0].numpy()
inp2 = inp[1].numpy()
C_so = ct.CDLL(os.path.abspath("test.so"))

out = np.zeros(2)
out_ptr = ct.c_void_p(out.ctypes.data)
C_so.test_fn(ct.c_double(inp1), ct.c_double(inp2), out_ptr)

import pdb
pdb.set_trace()