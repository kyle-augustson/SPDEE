# Requirements and capability tiers

SPDEE runs with three packages.  Everything else is an optional tier that
adds a core, a solver or a device, and its absence is reported, never
fatal: `python -m spdee.check` prints what the installation can run and
exits 0 when the minimum tier works.

## Tiers

`pip install -e .` installs the minimum; `pip install -e ".[dedalus,gpu-cuda]"`
adds tiers.  The extras are declared in `pyproject.toml`.

| tier (extra) | packages | adds |
|---|---|---|
| minimum (`dependencies`) | `numpy>=1.24`, `scipy>=1.10`, `sympy>=1.12`, Python >= 3.10 | the IR and `spdee.Problem`; the collocated, staggered (Arakawa-C) and sh_fd cores; `shift_invert` with `method='arpack'` and `factor` in `scipy` (SuperLU), `lapack` (dense host LU), `fplus` (pure-Python fallback); `dense(backend='numpy')`; `spdee.analysis` |
| `dedalus` | `dedalus>=3.0.5` (pulls mpi4py, h5py) | the spectral core `spdee.backends.dedalus` (shell and ball) |
| `gpu-cuda` | `cupy-cuda13x>=14.0`, `nvmath-python[cu13]>=0.8` | `factor='cudss'` (cuDSS); `dense(backend='cupy')` with the LU on the device and cusolverDnXgeev (CUDA >= 12.8) or `cp.linalg.eig`; `managed_memory()`; `factor='cupy'` as an explicit option (cupyx splu factors on the host and solves slower on the device) |
| `gpu-cuda12` | `cupy-cuda12x>=13.0`, `nvmath-python[cu12]>=0.8` | the same on a CUDA 12 runtime (Xgeev from 12.8) |
| `gpu-rocm` | `cupy-rocm-7-0>=14.0` (the wheel name carries the ROCm major; integrated RDNA 3.5 parts need a source build, see below) | the device dense LU of `dense(backend='cupy')` (chosen by `auto` on a discrete part), the device sparse matvec, `managed_memory()`, `factor='cupy'` as an explicit option; geev runs on the host; no cuDSS |
| `ext` | `setuptools`, `wheel`, `numpy` at build time; a C99 compiler; a CBLAS/BLAS/LAPACK to link (`libcblas`, `libblas`, `liblapack` as conda and the distributions name them; `BLAS_LIBS` overrides the FPLUS link line) | the FPLUS C extension (int64 LU with selectable pivoting, `factor='fplus'` at speed) and `method='bbjd'` (Jacobi-Davidson QZ); built in place by `spdee/solvers/build_ext.sh` |
| `viz` | `matplotlib>=3.7` | the figure scripts under `scripts/`, `tools/` and `examples/` |
| `mpi` | `mpi4py>=3.1` | the distributed resolution sweeps in `tools/` and `scripts/` |
| `test` | `pytest>=7` | `tests/`; the Dedalus, GPU and C-extension tests skip when their tier is absent |
| `all` | `dedalus`, `gpu-cuda`, `ext`, `viz`, `mpi` | a machine with an NVIDIA GPU.  ROCm is excluded because the CUDA and ROCm CuPy wheels conflict |

The version floors are conservative and were fixed by reading the API the
code uses: no numpy API newer than the `Generator` random interface
(1.17), no scipy API newer than `spmatrix`, `splu`/`spilu`/`eigs`/`svds`
and `spsolve_triangular` (all <= 1.10), generic sympy, and none of the
aliases numpy 2 removed.  The tree is tested at numpy 2.3, scipy 1.16,
sympy 1.14 on Python 3.13 (a conda env with the Dedalus recipe).  The C
extensions are compiled against the installed numpy headers.

Dedalus v3 ships no wheels and needs MPI, FFTW and HDF5 at build time;
`scripts/install_dedalus_env.sh` builds the whole conda environment
(Python, OpenBLAS numpy/scipy, OpenMPI, FFTW, parallel HDF5, mpi4py, h5py,
Dedalus from the upstream checkout) and is how the tested install was
made.  `pip install spdee[dedalus]` works only where those libraries are
already present.  Do not let MKL into a Dedalus process (see the switches
below).

## Vendor and device-class matrix

`spdee.solvers.gpu` probes the device once per process and each GPU path
consults it.  A capability that is missing degrades to the host path with
one `GPUFallbackWarning` naming the request, the reason and the fallback;
a capability that is present but fails numerically raises, so `auto` can
move on.  Automatic choices are decided by device **class**, not vendor:
`prefer_device()` is False on an integrated device (the `integrated`
field of `getDeviceProperties`, shares system memory) and True on a
discrete or datacenter device of either vendor; `dense(backend='auto')`
then needs a device dense LU and `factor='auto'` needs cuDSS.

| capability | CUDA (`gpu-cuda`) | ROCm (`gpu-rocm`) | no GPU |
|---|---|---|---|
| `gpu.vendor()` | `"cuda"` | `"rocm"` (from `cupy.cuda.runtime.is_hip`; librocsolver/librocblas mapped, no libcusolver or libhipsolver) | `None` |
| device class | `discrete`, or `integrated` (Jetson-like parts) | `discrete` (MI300-class) or `integrated` (APUs such as the Radeon 890M) | - |
| dense LU (`dense(backend='cupy')` transform) | cuSOLVER getrf/getrs through `cupyx.scipy.linalg.lu_factor` | rocSOLVER through the same call | host scipy LU |
| dense geev | `cusolverDnXgeev` (CUDA >= 12.8) through ctypes, else `cp.linalg.eig` (CuPy >= 14, probed by a trial call), else host | host LAPACK today: rocSOLVER has no nonsymmetric eigensolver and `cp.linalg.eig` exists but raises `geev is not available`; libcusolver is never loaded; MAGMA-HIP is the future route to a device geev | host LAPACK |
| `factor='cupy'` (cupyx splu) | explicit only: SuperLU factors on the host, the device solves are slower (33x measured on the 890M) | the same | SuperLU with one warning |
| `factor='cudss'` (cuDSS) | yes, with matching, behind the residual gate | no: NVIDIA-only, and CuPy offers no device sparse direct solver on either vendor; SuperLU with one warning | SuperLU with one warning |
| `factor='auto'` | cudss on a discrete GPU, then scipy | scipy | scipy |
| `dense(backend='auto')` | cupy on a discrete GPU, numpy on an integrated one | cupy on a discrete GPU (LU on the device, geev on the host, silently), numpy on an integrated one | numpy |
| explicit `backend='cupy'` | honoured; a `GPUPerformanceWarning` on an integrated device | honoured: LU on the device, geev on the host with one `GPUFallbackWarning`; a `GPUPerformanceWarning` on an integrated device | numpy with one warning |
| `managed_memory()` | `cudaMallocManaged` pool | `hipMallocManaged` through the same `cupy.cuda.malloc_managed` (constructs and allocates on HIP) | False with one warning |
| sparse CSR matvec on the device (matrix-free, bbjd/JDMF) | yes | yes (2.2x faster than the host at N = 30000 on the 890M) | host |
| `available()` | `gpu_vendor='cuda'`, `cupy`, `cudss`, `dense_cupy`, `gpu_geev` True | `gpu_vendor='rocm'`, `cupy`, `dense_cupy` True; `cudss`, `gpu_geev` False | all GPU keys False |

`available()` keeps its original keys (`cupy`: CuPy with a visible device
of any vendor; `cudss`: cuDSS can factor; `dense_cupy`: the census can run
on the device) and adds `gpu_vendor`, `gpu_device`, `gpu_dense_lu`,
`gpu_sparse_lu`, `gpu_geev`.  The reason behind a False is in
`gpu.features()`.

### AMD: two device classes

The ROCm column follows CuPy's ROCm build: same `cupy` namespace, HIP
mapped behind the `cupy.cuda` names, rocSOLVER behind
`cupyx.scipy.linalg.lu_factor`, no Xgeev, no cuDSS.  The AMD production
target is MI300-class hardware (CDNA3: MI300A/MI300X/MI325X, HBM3 with
128-192 GB, full rocSOLVER/rocSPARSE, MAGMA available, MI300A with unified
CPU-GPU memory); the correctness harness is an integrated Radeon 890M.
The two classes behave differently and the layer treats them by class.

**Integrated (Radeon 890M, gfx1150; Fedora 44, Ryzen AI 9 HX 370, ROCm
7.1.1 from the distribution, CuPy 14.2 built from source): a correctness
path.**  Measured there:

- `cupy.cuda.runtime.is_hip` is True, `getDeviceCount()` is 1 and the
  device name is `b'AMD Radeon 890M Graphics'`; `cupy.show_config()`
  reports the platform as AMD ROCm but lists hipSOLVER under cuSOLVER's
  name.  The probe reads the vendor from `is_hip` only, never from a
  library name; after import and an LU, `/proc/self/maps` shows
  librocsolver.so and librocblas.so, no libcusolver and no libhipsolver,
  so the ctypes probe for libcusolver cleanly finds nothing and the check
  prints the mapped libraries as a secondary signal.
- The PyPI `cupy-rocm-*` wheels ship code for a fixed list of discrete
  gfx targets and no gfx1150: the wheel segfaults on import there (which
  no probe can catch).  CuPy had to be built from source:
  `CUPY_INSTALL_USE_HIP=1 ROCM_HOME=/usr HCC_AMDGPU_TARGET=gfx1150
  pip install --no-binary cupy cupy`.  The `gpu-rocm` extra names the wheel
  for the discrete case; integrated RDNA 3.5 parts need the source build.
- rocSOLVER's blocked getrf makes different, equally valid pivot choices
  from LAPACK: on a well-conditioned 500 x 500 complex matrix 74 of the 500
  pivots differ and the packed factors differ by O(100), while `P L U - A`
  is 1.5e-13 and the solve residuals are 1.4e-12 on both, and the device
  and host solutions agree to 4e-13.  The pivot convention matches scipy
  (0-based int32).  Every test of the device LU therefore compares
  invariants (the P L U reconstruction, the solve residual against the
  pencil, the eigenvalues), never packed factors or pivot vectors:
  `tests/test_requirements.py::test_device_dense_lu_by_invariants`.
- The dense LU on the integrated device never wins: 0.52x host speed at
  N = 2000, 0.41x at 5000, 0.42x at 10000, no crossover within the 48 GiB
  framebuffer (a BIOS carveout of the 96 GiB; the check's device-memory
  line is meaningful there); 1.7 to 2 times slower than the 24-thread host
  on a 500 x 500 LU.  Hence `prefer_device()` is False on an integrated
  device and `auto` stays on the host; an explicit `backend='cupy'` is
  honoured with a `GPUPerformanceWarning`.
- `hasattr(cupy.linalg, "eig")` is True on ROCm but the call raises
  `RuntimeError('geev is not available')`.  The probe never gates on the
  attribute: `cupy_geev` is decided by a trial call on a 2 x 2 matrix on
  CUDA and marked unavailable on ROCm, and the census chain in
  `gpu_dense._geev` wraps the real call in try/except so a probe that is
  wrong still lands on host LAPACK (`test_geev_that_raises_reaches_host`).
- `cupyx.scipy.sparse.linalg.splu` factors on the host with scipy's SuperLU
  and moves only the triangular solves to the device (its own docstring
  says so): factor 4.34 s device vs 3.45 s host, solve 0.245 s vs 0.0074 s,
  33x slower solves, identical factorization.  This is CuPy's design on
  CUDA too, so `factor='cupy'` is never chosen automatically on any
  hardware and is kept as an explicit option whose docstring says what it
  does.
- The sparse CSR matvec on the device is 3.8x slower than the host at
  N = 3000 and 2.2x faster at N = 30000 (residual 4e-16): the one device
  win on that machine, and it is the matrix-free operation of bbjd/JDMF.
- `cupy.cuda.MemoryPool(cupy.cuda.malloc_managed)` constructs and allocates
  on HIP, so `managed_memory()` stays enabled on ROCm.

**Discrete / datacenter (MI300-class): expected to win on the dense LU
and the matvec, untested here.**  `prefer_device()` is True, so
`dense(backend='auto')` takes the LU on the device as on NVIDIA and the
geev runs on the host without a warning (the transform is O(N^3) with a
small constant and the host geev dominates; a device geev needs MAGMA-HIP,
not yet wired).  `factor='auto'` is SuperLU: CuPy offers no device sparse
direct solver on either vendor, and cuDSS is NVIDIA-only.  The device
sparse matvec is the path to build on for the matrix-free solvers.  When
such a machine is available, `python -m spdee.check` prints the device
class, memory and capability table so its numbers can be set against
those above.

`tests/test_requirements.py::test_rocm_simulation` exercises the ROCm
branch on a discrete CUDA machine by patching `is_hip` while the probe
runs: the vendor probe, the capability table, `auto` taking the device LU
with the geev on the host silently, the explicit census warning once, and
the promise that libcusolver is never touched.
`test_device_dense_lu_by_invariants`, `test_geev_that_raises_reaches_host`
and `test_integrated_device_auto_prefers_host` cover the measured facts
above.  What has not been run on AMD hardware is the full suite through
this layer; `python -m spdee.check` is written to be pasted back verbatim
from such a machine.  Run it, and the tests, with the interpreter's full
path (`/path/to/env/bin/python -m spdee.check`): a bare `python` may be
the system one.

The tiers need not coexist in one environment.  One development machine runs two:
a conda env (Python 3.13, numpy 2.5, the source-built ROCm CuPy, no
Dedalus), which exercises the minimum tier and the ROCm path and where
`backend='dedalus'` must raise the ImportError naming this document, and
a Dedalus venv (Python 3.14, numpy 2.4, scipy 1.17, Dedalus 3.0.5 at
upstream `2a2ad15`, no CuPy), which exercises the spectral core with the
GPU absent rather than switched off.  `python -m spdee.check` reports
either situation, and nothing in `tests/test_requirements.py` assumes
Dedalus and CuPy are installed together: the GPU tests skip without a
device, the hidden-Dedalus test hides it whether or not it is installed.
`requires-python` is `>=3.10` with no upper bound; the code uses nothing
removed in Python 3.14.

## Environment switches

| variable | effect |
|---|---|
| `SPDEE_NO_GPU=1` | the probe reports no device; `factor='auto'` and `backend='auto'` go to the host; explicit GPU requests degrade with one warning.  For CPU-only runs of the suite and for users who want the GPU left alone.  Read on every call, so it can be set from a test.  Note that the fast suite's dense censuses then run on the host: 1776 tests in about 84 minutes on 3 threads (test_analytic_evp alone 22 minutes) against about 15 with a discrete GPU |
| `CUDSS_USE_MATCHING=0` | restores cuDSS's default (no maximum-weight matching).  SPDEE enables matching (`use_matching` through nvmath 0.x, `matching_algorithm = MAX_DIAG_PRODUCT` from nvmath 1.0, whose default and count / min-diagonal choices are as wrong as none) because without it cuDSS returns wrong solves on pencils with zero diagonal entries; the residual gate in `spdee.solvers` then rejects the factorization and `auto` moves on |
| `CUDSS_THREADING_LIB` | passed to nvmath's `DirectSolverOptions` as the cuDSS threading layer |
| `SPDEE_USE_PARDISO=1` | the fd2 BVP direct solve uses pypardiso (MKL) instead of SuperLU.  Never in a Dedalus process: `libmkl_rt`'s FFTW symbols shadow the FFTW Dedalus links and every transform fails |
| `GPU_SI_HOST_TRANSPOSE=0` | `gpu_si`'s host adapter factors the matrix directly instead of its transpose |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | BLAS and OpenMP thread counts; keep them at 4 or less on a shared machine and at 1 for Dedalus |

## The check

```
$ python -m spdee.check            # the table below, exit 0 when the minimum tier works
$ python -m spdee.check --json     # the same report as JSON
$ SPDEE_NO_GPU=1 python -m spdee.check --no-solve
```

The report lists the package versions by tier, each core (importable or
why not), each solver method, factor and dense backend (available or why
not, and what `auto` resolves to), the GPU vendor, device, memory and
capability table, the thread settings and the switches, and then
compiles the m=2 inertial-mode problem on every importable host core and
finds the Rossby mode with ARPACK + SuperLU (and the dense host census on
the sh_fd pencil).  `spdee.check.report()` returns the same as a dict.

## ROCm baseline, measured

Fedora 44, Ryzen AI 9 HX 370 with the integrated Radeon 890M (gfx1150), ROCm 7.1.1
from Fedora's packages, CuPy 14.2 built from source against it, SPDEE at 4e2ea90.
Two environments, disjoint: a conda environment with the ROCm CuPy and no Dedalus,
and a Dedalus virtual environment on Python 3.14 with no CuPy.  Everything one
skips for a missing component the other runs, so between them every capability
on the machine is exercised.

| run | result |
|---|---|
| conda: requirements, solvers, examples | 29 passed, 5 skipped |
| conda: analytic benchmarks | 69 passed, 22 skipped (the Dedalus cases) |
| conda: fast suite, first half | 799 passed, 32 skipped |
| conda: fast suite, second half | 857 passed, 640 skipped (the slow coverage cells) |
| Dedalus venv: requirements | 7 passed, 4 skipped (the device cases) |

No failure in any of the five; 36 minutes of wall clock in all.  The device
routes to the host everywhere by the class rule, since on this part dense LU
runs at 0.4 to 0.5 of host speed at every size, the CuPy sparse factor factors
on the host regardless and solves 33 times slower on the device, and no device
nonsymmetric eigensolver exists; the sparse matvec is 2.2 times faster on the
device above ten thousand unknowns.  The m = 2 Rossby mode agrees to 1e-15
through every backend and factor path.  The integrated-device numbers are not
expectations for MI300-class hardware, which is the AMD target and untested.
