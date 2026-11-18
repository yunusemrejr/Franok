#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
clang --target=wasm32 -O3 -msimd128 -nostdlib runtime/kernels.c -o web/franok.wasm \
  -Wl,--no-entry -Wl,--export-memory -Wl,--export=__heap_base \
  -Wl,--export=gemv8 -Wl,--export=unpack8 -Wl,--export=gemv -Wl,--export=unpack -Wl,--export=rms -Wl,--export=add \
  -Wl,--export=rotary -Wl,--export=attend -Wl,--export=swiglu -ffreestanding
