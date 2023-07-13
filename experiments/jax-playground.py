import jax

def slow_f(x):
  return x * x + x * 2.0 # element-wise ops see a large benefit from fusion

x = jax.numpy.ones((5000, 5000))
fast_f = jax.jit(slow_f)
