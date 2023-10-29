#!/bin/bash

set -x

REPOSITORY_ROOT_DIR="${PWD}/$1"

BUILD_DIR="${REPOSITORY_ROOT_DIR}/build"

mkdir -p "${BUILD_DIR}"

cd "${BUILD_DIR}"

cmake -D USE_SUPERBUILD=OFF -D CMAKE_BUILD_TYPE:STRING=Release -S "${REPOSITORY_ROOT_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --config Release --parallel $(nproc)
