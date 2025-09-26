# Stepanov Library Benchmarks

This directory contains rigorous performance benchmarks validating the claims made in the Stepanov library whitepaper.

## Quick Start

```bash
# Compile and run the main benchmark
g++ -std=c++20 -O3 -DNDEBUG -march=native bench_realistic.cpp -o bench_realistic
./bench_realistic

# Optional: Build with CMake
mkdir build && cd build
cmake ..
make
./bench_realistic
```

## Benchmark Files

### `bench_realistic.cpp`
Main benchmark suite that tests all major performance claims:
- Diagonal matrix multiplication (claimed 1190x speedup)
- Symmetric matrix operations (claimed 50% memory, 2x speed)
- Triangular system solving (claimed 3x speedup)
- Expression templates (claimed 4x speedup)
- Power function complexity (O(log n) vs O(n))
- GCD algorithm comparisons

**Status**: ✓ Compiles and runs successfully

### `bench_fundamentals.cpp`
Tests core generic programming algorithms with custom types that satisfy library concepts.

**Status**: ⚠️ Requires fixes to library headers to compile

### `bench_eigen_comparison.cpp`
Compares against the Eigen library to provide industry baseline.

**Status**: ✓ Compiles but requires Eigen installation

## Key Findings

| Claim | Measured | Verdict |
|-------|----------|---------|
| Diagonal matrix 1190x | 212x | Overstated |
| Symmetric memory 50% | 50% | ✓ Verified |
| Symmetric speed 2x | 0.75x | False |
| Triangular solve 3x | 31x | ✓ Conservative |
| Expression templates 4x | 2.7x | Slightly overstated |
| Power O(log n) | ✓ | Verified |

## Compilation Requirements

- C++20 compatible compiler (GCC 10+, Clang 12+)
- Optional: Eigen3 library for comparison benchmarks
- Optional: Google Benchmark for advanced profiling

## Running Complete Suite

```bash
# Basic benchmarks
g++ -std=c++20 -O3 -DNDEBUG bench_realistic.cpp -o bench_realistic
./bench_realistic > results.txt

# With Eigen comparison (if installed)
g++ -std=c++20 -O3 -DNDEBUG bench_eigen_comparison.cpp -o bench_eigen
./bench_eigen >> results.txt

# View results
cat results.txt
```

## Reproducing Results

For most accurate results:

1. **Disable CPU frequency scaling**:
   ```bash
   sudo cpupower frequency-set -g performance
   ```

2. **Run multiple times** to account for variance:
   ```bash
   for i in {1..5}; do ./bench_realistic; done
   ```

3. **Use release builds** with full optimization:
   ```bash
   g++ -std=c++20 -O3 -DNDEBUG -march=native bench_realistic.cpp
   ```

## Interpreting Results

The benchmarks measure:
- **Absolute time**: How long operations take
- **Relative speedup**: Optimized vs naive implementations
- **Complexity verification**: Confirming O(n) vs O(log n) claims

Remember that:
- Exact speedups vary by hardware
- Cache effects significantly impact results
- Generic programming has inherent overhead (10-50%)
- The library's value is in correctness and flexibility, not just speed

## Adding New Benchmarks

To add new benchmarks:

1. Follow the existing pattern in `bench_realistic.cpp`
2. Include warmup iterations
3. Measure multiple samples for statistical significance
4. Compare against both naive and standard implementations
5. Document expected vs actual results

## Full Report

See `PERFORMANCE_VALIDATION_REPORT.md` for detailed analysis of all claims and findings.