import jax
jax.config.update("jax_enable_x64", True) # use float64 -- jax by default uses float32

def slow_f(x):
  return x * x + x * 2.0 # element-wise ops see a large benefit from fusion

x = jax.numpy.ones((5000, 5000))
fast_f = jax.jit(slow_f)

