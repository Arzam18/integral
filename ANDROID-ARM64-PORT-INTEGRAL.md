# Integral v8 — Android ARM64 NEON and DotProd Port

This port is based on the newer upstream Integral source. The older Android tree was used only as a reference for Android-specific build requirements; its older engine/search changes and its deleted SIMD implementations were not copied over the upstream source.

## Build variants

The GitHub Actions workflow builds two independent `arm64-v8a` executables:

- **NEON baseline:** `-march=armv8-a -mtune=cortex-a76`
- **NEON + DotProd:** `-march=armv8.2-a+dotprod -mtune=cortex-a76`

Both use Android NDK r27c (`27.2.12479018`), Android API 24, C++20, static libc++ (`ANDROID_STL=c++_static`), and `USE_NUMA=OFF`. The workflow has no `uci_date` input. It uploads engine-only binaries and SHA-256 files as separate artifacts.

The DotProd artifact requires an ARM64 device with the ARMv8.2 DotProd extension. The baseline NEON artifact is the compatibility build for ordinary AArch64 NEON devices.

## Network layout

The newer upstream CMake source selects the `yveltal` network by default. The workflow follows that source selection and downloads `yveltal.nnue` before preprocessing it.

The network must be preprocessed on the host for the AArch64 NEON layout. The host preprocessor is compiled with `BUILD_AARCH64_LAYOUT=1`: this selects 16-byte vectors, 32 NEON vector registers, and the AArch64 pack order without including ARM-only intrinsics. The resulting network is then embedded unchanged in both Android executables. The Android cross-build does not execute a target-architecture preprocessor.

## SIMD and NNUE changes

### NEON

The upstream `shared/simd.h` already contained several NEON implementations. Those implementations were retained, rather than importing the older Android fork's deletions. The Android target selection now activates the existing NEON paths explicitly.

### DotProd U8 × I8 arithmetic

Integral's feature-transformer output is packed to unsigned 8-bit values and can reach 255 (`kFtQuantization` is 255). A plain signed `SDOT` would therefore be wrong for inputs above 127.

The DotProd implementation XOR-biases each U8 input with `0x80`, reinterprets that byte as signed, performs `SDOT`, and adds `128 * sum(weights)` for each four-byte dot-product lane. This is exact for the full U8 input range and the signed I8 weight range; it does not change network weights or quantization.

### Sparse NNUE feature indices

The NNUE sparse path treats each I32 lane as a group of four packed U8 features. The nonzero mask now detects a lane that is *not zero*, rather than testing whether the reinterpreted signed I32 value is positive. This matters when a feature byte has its high bit set.

Only active four-byte groups are appended to the index array. Its capacity is `kL1Size / 4`, which is the maximum possible number of such groups across both perspectives. This avoids the old lookup approach writing a full SIMD batch for a mask containing fewer active groups.

### Android aligned allocation

Android uses `posix_memalign` for aligned table/network allocations, with the matching `std::free` release path. `sys/mman.h` is included for Android page allocation functions.

## Source files changed

- `.github/workflows/android.yml` — NDK setup, network download, host AArch64-layout preprocessing, two Android build variants, ELF/SIMD checks, and artifacts.
- `CMakeLists.txt` — Android options, architecture selection, preprocessing handoff, and Android LTO handling.
- `preprocess/CMakeLists.txt` — host AArch64-layout preprocessing mode.
- `shared/simd.h` — Android NEON selection, host layout mode, exact DotProd U8×I8 operation, and correct nonzero-lane masks.
- `src/engine/evaluation/nnue/nnue.cc` — safe active-group sparse index collection.
- `src/utils/hash_table.h` — Android POSIX aligned allocation and memory-mapping header.
- `ANDROID-ARM64-PORT-INTEGRAL.md` — this document.
- `ANDROID-ARM64-PORT-INTEGRAL-CHANGELOG.txt` — per-file change log.

## Validation performed

- Full host x86-64 AVX2 configure, compile, and link completed using a zero-filled fixture network with the same raw file size as the source network.
- The host-built engine passed a UCI startup/readiness and three-ply search smoke test with that fixture network. This checks executable integration only; the zero-filled fixture is not a usable playing network and proves nothing about playing strength.
- The host AArch64-layout preprocessor built and processed the same-size fixture network. Its output was exactly 91,255,328 bytes, matching the expected `Network` layout size for 16-byte AArch64 vectors.
- AArch64 baseline NEON helper code cross-compiled with Clang and produced NEON instructions without `SDOT`.
- AArch64 DotProd helper code cross-compiled with Clang and the disassembly contained `SDOT`.
- The unsigned-byte centering/correction formula matched scalar U8×I8 arithmetic for 100,000 randomized vectors.
- AVX2 `NonZeroMask` matched scalar raw-I32 nonzero masks for 100,000 randomized vectors, including negative signed lane representations.
- The sparse-index capacity calculation was checked at the maximum 256 active groups.
- The Android `posix_memalign` code path compiled and passed a 2 MiB alignment/allocation/free test on the host POSIX runtime.
- Workflow YAML, CMake option override logic, ZIP integrity, and archive consistency were checked.

## Not yet verified

A full Android NDK build against the actual downloaded `yveltal.nnue` network, Android device runtime, device NPS, and playing strength have not yet been verified in this environment. The workflow is intended to perform the real Android cross-build and checks the resulting AArch64 ELF, interpreter, static C++ runtime, and required SIMD instructions. The network download URL itself was not independently downloaded in this environment; the workflow fails explicitly if the release asset is unavailable or the response is not a network file.
