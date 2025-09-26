# Stepanov Compression Benchmark Suite

## Overview

An elegant, informative benchmarking system for compression algorithms that explores the deep connections between compression, learning, and intelligence. This suite provides comprehensive metrics, beautiful visualizations, and philosophical insights into the nature of compression.

## Features

### Core Metrics
- **Compression Ratio**: Original size / compressed size
- **Compression Speed**: MB/s throughput
- **Decompression Speed**: MB/s throughput
- **Memory Usage**: Peak and average memory consumption
- **CPU Utilization**: Per-core usage for parallel algorithms

### Advanced Metrics
- **Kolmogorov Complexity Approximation**: K(x) ≈ |C(x)|
- **Entropy Estimation**: Information content in bits per byte
- **Incompressibility Detection**: Identifies random/incompressible data
- **Compression Stability**: Variance across similar inputs
- **Progressive Quality**: For lossy compression algorithms

### Intelligent Analysis
- **Data Characterization**: Identifies patterns and structure
- **Algorithm Recommendation**: Suggests best algorithm for data type
- **Worthiness Detection**: Determines when compression isn't beneficial
- **Pipeline Optimization**: Finds optimal algorithm combinations

## Quick Start

### Building

```bash
# Build everything
make -f Makefile.benchmark all

# Run tests
make -f Makefile.benchmark test

# Run demonstration
make -f Makefile.benchmark demo
```

### Basic Usage

```cpp
#include <stepanov/compression/benchmark.hpp>

using namespace stepanov::compression::benchmark;

int main() {
    // Create benchmark suite
    benchmark_suite suite;

    // Add compression algorithms
    suite.add_algorithm(your_compressor())
         .add_algorithm(another_compressor());

    // Add test data
    suite.add_corpus(canterbury_corpus())
         .add_corpus(text_corpus());

    // Configure and run
    auto results = suite.run();

    // Display results
    results.print_summary();
    results.recommend_best();
    results.export_csv("results.csv");
}
```

## API Design

### Compression Algorithm Interface

Any compression algorithm can be benchmarked by satisfying the `CompressorConcept`:

```cpp
template<typename T>
concept CompressorConcept = requires(T t, const std::vector<uint8_t>& input) {
    { t.compress(input) } -> std::convertible_to<std::vector<uint8_t>>;
    { t.decompress(input) } -> std::convertible_to<std::vector<uint8_t>>;
    { t.name() } -> std::convertible_to<std::string>;
    { t.description() } -> std::convertible_to<std::string>;
};
```

### Test Corpus Management

Create custom test corpora:

```cpp
test_corpus corpus("MyData", "custom");
corpus.add_text_sample("Sample text data");
corpus.add_sample(binary_data);

// Corpus automatically calculates:
// - Entropy
// - Redundancy
// - Total size
// - File count
```

### Fluent Interface

The suite uses a fluent, chainable API:

```cpp
benchmark_suite()
    .add_algorithm(lz77_compressor())
    .add_algorithm(huffman_compressor())
    .add_corpus(your_corpus())
    .verbose(true)
    .runs_per_test(10)
    .run()
    .print_summary();
```

## Visualization

The suite produces beautiful ASCII art visualizations:

```
┌─── Compression-Speed Tradeoff ───┐
│ Speed                             │
│ (MB/s)                           │
│ 1000 ┤                           │
│      │        B                  │
│      │                           │
│      │    A       C              │
│      │                           │
│      │                D          │
│    0 ┤                           │
│     └────┴────┴────┴────┴────┤
│     0   2   4   6   8   10     │
│         Compression Ratio       │
└─────────────────────────────────┘
```

## Theoretical Foundation

The benchmark suite is built on deep theoretical principles:

### Kolmogorov Complexity
The shortest description length of data approximates its algorithmic complexity:
```cpp
K(x) ≈ |compress(x)|
```

### Shannon Entropy
Information-theoretic limit for lossless compression:
```cpp
H(X) = -Σ p(xi) log₂ p(xi)
```

### Minimum Description Length
Best model minimizes:
```cpp
MDL = L(Model) + L(Data|Model)
```

### Compression-Based Similarity
Normalized Compression Distance for clustering:
```cpp
NCD(x,y) = (C(xy) - min(C(x),C(y))) / max(C(x),C(y))
```

## Standard Test Corpora

### Canterbury Corpus
Mixed real-world data including text, source code, and binary files.

### Text Corpus
Literary text, source code, and documentation.

### Synthetic Corpus
- Highly compressible (repeated patterns)
- Incompressible (random data)
- Semi-compressible (patterns with noise)

### Rich Mix Corpus
Comprehensive test including Shakespeare, JSON, XML, and binary data.

## Insights from Benchmarking

The suite reveals profound insights:

1. **Compression = Understanding**: Algorithms that "understand" data structure compress better
2. **No Free Lunch**: No single algorithm dominates all data types
3. **Speed-Compression Tradeoff**: Deeper understanding requires more computation
4. **Pattern Universality**: All compressible data contains patterns
5. **Kolmogorov in Practice**: Compression ratios approximate algorithmic complexity

## Advanced Features

### Statistical Rigor
- Multiple runs per test
- Confidence intervals
- Standard deviation tracking
- Outlier detection

### Memory Profiling
- Peak memory tracking
- Average usage calculation
- Algorithm-specific estimates

### Export Capabilities
- CSV export for analysis
- JSON export (extensible)
- Comparative reports

### Parallel Benchmarking
- Thread-safe design
- Parallel corpus testing
- CPU utilization tracking

## Philosophy

> "Compression is not about making things smaller. It's about understanding what's essential."

This benchmark suite embodies the principle that compression, learning, prediction, and intelligence are fundamentally equivalent. By measuring compression, we measure understanding itself.

## Examples

### Compare Algorithm Families

```cpp
// Compare entropy coders
suite.add_algorithm(huffman_coder())
     .add_algorithm(arithmetic_coder())
     .add_algorithm(range_coder());

// Compare dictionary methods
suite.add_algorithm(lz77())
     .add_algorithm(lz78())
     .add_algorithm(lzw());

// Compare modern methods
suite.add_algorithm(brotli())
     .add_algorithm(zstd())
     .add_algorithm(lzma());
```

### Custom Metrics

```cpp
class custom_metrics : public compression_metrics {
    double perplexity;
    double compression_time_variance;

    void calculate(const results& r) {
        // Custom metric calculations
    }
};
```

### Compression-Based Clustering

```cpp
auto similarity_matrix = calculate_ncd_matrix(documents);
auto clusters = hierarchical_clustering(similarity_matrix);
print_dendrogram(clusters);
```

## Requirements

- C++20 compiler with concepts support
- Standard library with `<ranges>` support
- Optional: OpenMP for parallel benchmarking

## Integration

The benchmark suite integrates seamlessly with the Stepanov library:

```cpp
#include <stepanov/compression/kolmogorov.hpp>
#include <stepanov/compression/adaptive.hpp>
#include <stepanov/compression/benchmark.hpp>

// Use advanced compressors from the library
suite.add_algorithm(kolmogorov_optimal_compressor())
     .add_algorithm(adaptive_markov_compressor());
```

## Future Directions

- **Quantum Compression**: Benchmarking quantum algorithms
- **Neural Compression**: Deep learning-based methods
- **Semantic Compression**: Understanding-based compression
- **Real-time Visualization**: Live compression monitoring

## References

- Shannon, C.E. (1948). "A Mathematical Theory of Communication"
- Kolmogorov, A.N. (1965). "Three Approaches to the Definition of Information"
- Li, M., & Vitányi, P. (2008). "An Introduction to Kolmogorov Complexity"
- Hutter, M. (2005). "Universal Artificial Intelligence"

## License

Part of the Stepanov Generic Programming Library.

---

*"To measure is to understand. To compress is to comprehend."*