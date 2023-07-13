from jax import jit
from jax import numpy as jnp

def slow_f(x):
  return x * x + x * 2.0 # element-wise ops see a large benefit from fusion

x = jnp.ones((5000, 5000))
fast_f = jit(slow_f)

fast_f(x) 
slow_f(x) 