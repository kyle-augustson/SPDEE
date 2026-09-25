#!/usr/bin/env bash
# Build the C extensions of the optional solver backends in place.
#   SPARSE_LU_OPENMP=1 (default) enables OpenMP in FPLUS; SPARSE_LU_BLAS=1
#   (default) links the CBLAS kernels of the supernodal update (BLAS_LIBS
#   overrides the link line, e.g. BLAS_LIBS="-lopenblas").
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python}"
echo "== FPLUS"; (cd "$HERE/fplus" && rm -rf build && SPARSE_LU_OPENMP="${SPARSE_LU_OPENMP:-1}" SPARSE_LU_BLAS="${SPARSE_LU_BLAS:-1}" "$PY" setup.py build_ext --inplace -q)
echo "== bbjd";  (cd "$HERE/bbjd"  && rm -rf build && "$PY" setup.py build_ext --inplace -q)
echo "done"
