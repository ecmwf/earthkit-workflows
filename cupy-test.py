import numpy as np
import cupy as cp

x_gpu = cp.array([1, 2, 3])

x_cpu = np.array([1, 2, 3])
l2_cpu = np.linalg.norm(x_cpu)

x_gpu = cp.array([1, 2, 3])
l2_gpu = cp.linalg.norm(x_gpu)


x_cpu = np.array([1, 2, 3])
x_gpu = cp.asarray(x_cpu)  # move the data to the current device.

x_gpu = cp.array([1, 2, 3])  # create an array in the current device
x_cpu = cp.asnumpy(x_gpu)  # move the array to the host.



# automatically use GPU if arguments are on the GPU
# Stable implementation of log(1 + exp(x))
def softplus(x):
    xp = cp.get_array_module(x)  # 'xp' is a standard usage in the community
    print("Using:", xp.__name__)
    return xp.maximum(0, x) + xp.log1p(xp.exp(-abs(x)))

softplus(x_gpu)


# custom kernels

squared_diff = cp.ElementwiseKernel(
   'float32 x, float32 y',
   'float32 z',
   'z = (x - y) * (x - y)',
   'squared_diff')


x = cp.arange(10, dtype=np.float32).reshape(2, 5)
y = cp.arange(5, dtype=np.float32)
z = squared_diff(x, y)

print(z)


# reduce kernel

l2norm_kernel = cp.ReductionKernel(
    'T x',  # input params
    'T y',  # output params
    'x * x',  # map
    'a + b',  # reduce
    'y = sqrt(a)',  # post-reduction map
    '0',  # identity value
    'l2norm'  # kernel name
)
x = cp.arange(10, dtype=np.float32).reshape(2, 5)
y = l2norm_kernel(x, axis=1)
print(y)


# Raw kernels

add_kernel = cp.RawKernel(r'''
extern "C" __global__
void my_add(const float* x1, const float* x2, float* y) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    y[tid] = x1[tid] + x2[tid];
}
''', 'my_add')
x1 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
x2 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
y = cp.zeros((5, 5), dtype=cp.float32)
add_kernel((5,), (5,), (x1, x2, y))  # grid, block and arguments
print(y)

print("memory pointer:")
print(y.data.ptr, y.nbytes)