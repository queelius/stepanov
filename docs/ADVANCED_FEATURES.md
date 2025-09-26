# Stepanov Library - Advanced Features

This document describes the cutting-edge features that make the Stepanov library unique in the C++ ecosystem.

## 1. Tropical Mathematics (`stepanov::tropical`)

Tropical mathematics transforms nonlinear problems into linear ones by redefining addition and multiplication. This elegant mathematical framework has applications in optimization, machine learning, and computational biology.

### Features
- **Min-Plus Algebra**: Addition becomes minimum, multiplication becomes addition
- **Max-Plus Algebra**: Addition becomes maximum, multiplication becomes addition
- **Tropical Matrix Operations**: Matrix multiplication computes shortest/longest paths
- **Tropical Polynomials**: Piecewise linear functions with applications in optimization
- **Applications**: Shortest paths, scheduling, Viterbi algorithm, auction theory

### Example
```cpp
// Solve all-pairs shortest paths with one line of tropical algebra
auto all_paths = adjacency_matrix.kleene_star();

// Optimal scheduling with max-plus algebra
scheduler.compute_schedule(precedence_graph, task_durations);
```

### Innovation
Rarely seen in C++ libraries, tropical mathematics provides elegant solutions to graph algorithms and dynamic programming problems that are typically solved with complex imperative code.

## 2. Succinct Data Structures (`stepanov::succinct`)

Space-optimal data structures that achieve information-theoretic lower bounds while maintaining fast query times.

### Features
- **Bit Vector with Rank/Select**: O(1) rank and select with only o(n) extra space
- **Wavelet Tree**: Compressed sequence representation with range queries
- **FM-Index**: Full-text search on compressed text (like grep on compressed files)
- **Succinct Trees**: Trees in 2n + o(n) bits with full navigation

### Example
```cpp
// Search compressed text without decompression
fm_index compressed_genome(dna_sequence);
auto occurrences = compressed_genome.locate("ACGT");

// Range queries on compressed sequences
wavelet_tree wt(data);
auto median = wt.quantile(left, right, k);
```

### Innovation
These structures are at the forefront of theoretical computer science, rarely implemented in production C++ libraries.

## 3. Advanced Compression (`stepanov::compression`)

State-of-the-art compression algorithms that outperform traditional methods.

### Features
- **Asymmetric Numeral Systems (ANS)**:
  - rANS: Fast range variant
  - tANS: Table variant for optimal compression
  - Better than arithmetic coding, faster than Huffman
- **Context Mixing**: PAQ-style compression, best-in-class ratios
- **Prediction by Partial Matching (PPM)**: Adaptive context modeling
- **Burrows-Wheeler Transform**: Foundation of bzip2, with MTF encoding

### Example
```cpp
// Modern entropy coding that beats arithmetic coding
rans_encoder encoder;
encoder.initialize(symbol_frequencies);
auto compressed = encoder.encode(data);

// Context mixing for maximum compression
context_mixer mixer(num_models);
auto prediction = mixer.predict(contexts, symbol);
```

### Innovation
ANS is the future of compression, used by Facebook's Zstandard and Apple's LZFSE, but rarely available as a reusable C++ component.

## 4. Cache-Oblivious Algorithms (`stepanov::cache_oblivious`)

Algorithms that achieve optimal cache performance on any hardware without tuning.

### Features
- **Van Emde Boas Layout**: Trees with optimal cache complexity
- **Cache-Oblivious Matrix Multiplication**: Recursive blocking that adapts to any cache
- **Funnel Sort**: Optimal sorting for any memory hierarchy
- **Cache-Oblivious B-Tree**: Database index with automatic cache adaptation

### Example
```cpp
// Matrix multiply that's fast on any hardware
cache_oblivious::matrix A(1000, 1000);
auto C = A * B;  // Automatically optimized for L1, L2, L3 cache

// Tree that's cache-efficient without knowing cache size
veb_tree<int> tree;
tree.range_query(low, high);  // O(log_B n) I/Os
```

### Innovation
These algorithms represent the pinnacle of algorithm design, achieving theoretical optimality without hardware-specific tuning.

## 5. Formal Verification (`stepanov::verify`)

Correctness tools inspired by formal methods and property-based testing.

### Features
- **Design by Contract**: Preconditions, postconditions, invariants with zero-cost in release
- **Refinement Types**: Types with predicates (positive numbers, probabilities)
- **Property-Based Testing**: QuickCheck-style testing with shrinking
- **Bounded Model Checking**: Exhaustive verification up to bounded depth
- **Stateful Testing**: Model-based testing for complex systems

### Example
```cpp
// Self-documenting, self-checking code
auto divide(double a, double b) -> double {
    REQUIRES(b != 0);
    double result = a / b;
    ENSURES(std::abs(result * b - a) < epsilon);
    return result;
}

// Types that enforce invariants
positive_t<int> count(42);  // Can never be negative
probability_t<double> p(0.7);  // Always in [0, 1]

// Find bugs with generated test cases
property_test<int, int> test("commutativity",
    [](int a, int b) { return f(a, b) == f(b, a); });
```

### Innovation
Brings formal methods to practical C++ development, making verification accessible to everyday programming.

## Why These Features Matter

1. **Mathematical Elegance**: Solutions that are beautiful and correct by construction
2. **Performance**: State-of-the-art algorithms that outperform traditional approaches
3. **Space Efficiency**: Data structures at information-theoretic limits
4. **Correctness**: Built-in verification prevents bugs before they happen
5. **Innovation**: Features you won't find in Boost, STL, or other C++ libraries

## Use Cases

- **Bioinformatics**: Compress and search genomic data with FM-index
- **Machine Learning**: Tropical algebra for neural network verification
- **Databases**: Cache-oblivious B-trees for indices
- **Networking**: Shortest path routing with tropical matrices
- **Compression**: ANS for video codecs and file compression
- **Finance**: Verified numerical algorithms with contracts
- **Gaming**: Space-efficient data structures for large worlds

## Getting Started

```cpp
#include <stepanov/tropical.hpp>
#include <stepanov/succinct.hpp>
#include <stepanov/compression.hpp>
#include <stepanov/cache_oblivious.hpp>
#include <stepanov/verify.hpp>

// Build with C++20
// g++ -std=c++20 -O3 your_program.cpp
```

## Requirements

- C++20 compiler (GCC 11+, Clang 13+, MSVC 2022+)
- Header-only library, no dependencies
- Optional: Enable contracts with `-DCONTRACTS_ENABLED`

## Performance

All implementations are optimized for production use:
- Zero-cost abstractions through templates
- Cache-conscious memory layouts
- SIMD-friendly data structures where applicable
- Compile-time optimization opportunities

## References

- Tropical Mathematics: Maclagan & Sturmfels, "Introduction to Tropical Geometry"
- Succinct Structures: Navarro, "Compact Data Structures"
- ANS: Duda, "Asymmetric Numeral Systems"
- Cache-Oblivious: Frigo et al., "Cache-Oblivious Algorithms"
- Formal Methods: Meyer, "Object-Oriented Software Construction"

## License

These implementations follow Alex Stepanov's philosophy of generic programming and are provided for educational and practical use.