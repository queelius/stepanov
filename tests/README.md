# Test Organization

This directory contains all tests for the Stepanov generic mathematics library.

## Directory Structure

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests for feature combinations
├── benchmarks/        # Performance benchmarks
├── examples/          # Example programs and demonstrations
└── optimization/      # Optimization-specific tests
```

## Test Categories

### Unit Tests (`unit/`)
Core functionality tests for individual components:
- `test_math.cpp` - Basic mathematical operations
- `test_concepts.cpp` - Concept validation
- `test_gcd.cpp` - GCD algorithms
- `test_matrix_expressions.cpp` - Matrix expression templates
- `test_matrix_algorithms.cpp` - Matrix algorithm implementations
- `test_polynomial.cpp` - Polynomial arithmetic
- `test_bounded_nat.cpp` - Bounded natural numbers
- And more...

### Integration Tests (`integration/`)
Tests that verify components work together:
- `test_all_components.cpp` - Comprehensive integration test
- `test_allocators.cpp` - Memory allocator integration
- `test_compression*.cpp` - Compression algorithm tests
- `test_boolean.cpp` - Boolean algebra operations
- `test_geometry.cpp` - Geometric algorithms
- `test_iterators_ranges.cpp` - Iterator and range tests
- `test_matrix.cpp` - Matrix operations
- `test_innovations.cpp` - Innovative features showcase

### Benchmarks (`benchmarks/`)
Performance measurements and comparisons:
- `test_benchmark.cpp` - General performance benchmarks
- Algorithm-specific benchmarks

### Examples (`examples/`)
Demonstration programs showing library usage:
- `test_simple_demo.cpp` - Simple usage examples
- `test_advanced.cpp` - Advanced feature demonstrations
- `test_advanced_features.cpp` - Complex feature combinations

### Optimization Tests (`optimization/`)
Tests for optimization algorithms and techniques.

## Running Tests

### Run All Tests
```bash
cd build
ctest
```

### Run Specific Category
```bash
# Unit tests only
ctest -R "test_.*" --test-dir tests/unit

# Integration tests
ctest -R "test_.*" --test-dir tests/integration

# Benchmarks
ctest -R "test_.*" --test-dir tests/benchmarks
```

### Run Individual Test
```bash
./tests/unit/test_math
./tests/integration/test_compression
```

## Adding New Tests

1. **Unit Test**: Add to `unit/` if testing a single component in isolation
2. **Integration Test**: Add to `integration/` if testing multiple components together
3. **Benchmark**: Add to `benchmarks/` if measuring performance
4. **Example**: Add to `examples/` if demonstrating usage patterns

## Test Naming Convention

- Unit tests: `test_<component>.cpp`
- Integration tests: `test_<feature>_integration.cpp` or `test_<feature>.cpp`
- Benchmarks: `benchmark_<algorithm>.cpp`
- Examples: `example_<feature>.cpp` or `demo_<concept>.cpp`

## Coverage

To generate coverage reports:
```bash
cmake -DSTEPANOV_ENABLE_COVERAGE=ON ..
make
make test
make coverage
```

Coverage reports will be generated in `build/coverage_report/`.