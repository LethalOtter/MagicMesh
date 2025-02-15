import ctypes
import math as m

lib = ctypes.CDLL("./sharedlib.so")

lib.compute_sin.argtypes = [ctypes.c_double]
lib.compute_sin.restype = ctypes.c_double

lib.compute_cos.argtypes = [ctypes.c_double]
lib.compute_cos.restype = ctypes.c_double

lib.run.restype = ctypes.c_double


print(lib.compute_sin(m.pi))
print(lib.compute_cos(m.pi))
print(lib.run())
