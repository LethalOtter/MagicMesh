import ctypes
import math as m

# Load the shared library
lib = ctypes.CDLL("./sharedlib.so")  # Use "mathfuncs.dll" on Windows

# Define function argument and return types
lib.compute_sin.argtypes = [ctypes.c_double]
lib.compute_sin.restype = ctypes.c_double

lib.compute_cos.argtypes = [ctypes.c_double]
lib.compute_cos.restype = ctypes.c_double

lib.run.restype = ctypes.c_double


# Call functions
print(lib.compute_sin(m.pi))  # Should print sin(1.0)
print(lib.compute_cos(m.pi))  # Should print cos(1.0)
print(lib.run())
