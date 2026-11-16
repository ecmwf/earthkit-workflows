# base64 vs pybase64 vs fastpybase64

## Motivation
We base64-encode cloudpickled payloads (callables, types, job results in the gateway, logging config).
Results can be large, so we checked whether a faster base64 implementation is worthwhile.

## Experiment
* payload: `cloudpickle.dumps(np.random.rand(512MiB / 8))`, ie, roughly 512 MiB of bytes
* operation: `b64encode(payload)`, mean of 3 repetitions, one fresh process per library
* memory: `ru_maxrss` delta before and after encoding (see caveat below)
* machine: a linux x86_64 dev box, python 3.14, pybase64 1.5.2, fastpybase64 0.1.4; one run only, so rough numbers

Script (condensed; run with `uv run --with numpy --with cloudpickle --with pybase64 --with fastpybase64 python b.py <base64|pybase64|fastpybase64>`):

```python
import sys, time, resource, base64, numpy as np, cloudpickle
which = sys.argv[1]
a = np.random.rand(512*1024*1024//8)
data = cloudpickle.dumps(a); del a
if which == "pybase64":
    import pybase64; f = pybase64.b64encode
elif which == "fastpybase64":
    import fastpybase64; f = fastpybase64.b64encode
else:
    f = base64.b64encode
import gc; gc.collect()
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t = time.perf_counter()
for _ in range(3):
    out = f(data); n = len(out); del out
dt = (time.perf_counter() - t) / 3
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(which, f"{dt:.3f}s", f"out {n/2**20:.0f}MB", f"peak-delta {(peak-base)/1024:.0f}MB")
```

## Results

| library       | encode time | output size |
|---------------|-------------|-------------|
| base64        | 0.88 s      | 683 MB      |
| pybase64      | 0.43 s      | 683 MB      |
| fastpybase64  | 0.43 s      | 683 MB      |

* pybase64 and fastpybase64 are about 2x faster than stdlib, and equal to each other
* memory: all three allocate one output buffer of 4/3 of the input (683 MB); there is no meaningful difference
  between them. The measured `ru_maxrss` delta (~170 MB each) is not trustworthy, since the baseline peak already
  included the pickling phase -- treat the true overhead as "the output size" for all three
* not measured: decoding, `memoryview` inputs, other architectures

## Decision
Migrated to pybase64: same speed as fastpybase64, much more established, API-compatible with stdlib
(`b64encode`/`b64decode`), and output is byte-identical so existing encoded values remain readable.

## Portability assessment (PyPI, checked for pybase64 1.5.2)
* license BSD-2-Clause; `requires-python >=3.9` (we need >=3.11); 223 files in the release
* prebuilt wheels exist for CPython 3.9 to 3.15 (incl. free-threaded 3.13t+/3.14t) for every target we care about:

| target             | wheel tag (for cp311 to cp314)                                   |
|--------------------|------------------------------------------------------------------|
| linux x86_64       | `manylinux_2_17_x86_64` (manylinux2014) and `musllinux_1_2_x86_64` |
| linux aarch64      | `manylinux_2_17_aarch64` (manylinux2014) and `musllinux_1_2_aarch64` |
| macOS x86_64       | `macosx_10_9_x86_64` (cp311), `macosx_10_13_x86_64` (cp312, 313), `macosx_10_15_x86_64` (cp314+) |
| macOS arm64        | `macosx_11_0_arm64`                                              |

* also published: linux i686/armv7l/ppc64le/s390x/riscv64, windows (x86, x64, arm64), PyPy, GraalPy, android, iOS, wasm
* an sdist is available as fallback (C extension, so would need a compiler), but all four target platforms have wheels,
  so no compiler is needed for installs
* the library has pure-python fallback paths only for the API; the C extension is the fast path and it uses
  runtime SIMD dispatch (SSSE3/SSE4/AVX/AVX2 on x86, NEON on aarch64), so there is no build-time CPU requirement
* the minimal glibc is 2.17 (very old), minimal macOS is 10.9 to 10.15 depending on the python version, 11.0 for arm64
* fastpybase64 for comparison: 0.1.4, python >=3.10, wheels for the same four targets, but a much younger project
* caveat: the table is derived from PyPI file listing, we did not actually install on macOS or aarch64 machines
