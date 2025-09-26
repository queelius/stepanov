# Stepanov Library - Advanced Compression Algorithms

## Overview

This compression framework goes beyond traditional compression libraries by treating compression as a fundamental computational operation. It implements cutting-edge algorithms that demonstrate deep connections between compression, information theory, machine learning, and computation theory.

## Module Structure

```
compression/
├── kolmogorov.hpp      # Kolmogorov complexity approximations
├── grammar.hpp         # Grammar-based compression algorithms
├── adaptive.hpp        # Online and adaptive compression
├── specialized.hpp     # Domain-specific compressors
└── homomorphic.hpp     # Compression with homomorphic properties
```

## Key Innovations

### 1. Kolmogorov Complexity Approximations (`kolmogorov.hpp`)

**Fundamental Insight**: While Kolmogorov complexity is uncomputable, practical approximations provide universal measures of information content and similarity.

#### Algorithms:
- **Lempel-Ziv Complexity**: Measures randomness by counting minimal patterns
- **Normalized Compression Distance (NCD)**: Universal similarity metric
- **Minimum Description Length (MDL)**: Principled model selection
- **Algorithmic Mutual Information**: Information shared between objects
- **Bennett's Logical Depth**: Computational complexity measure
- **Sophistication Measure**: Length of shortest "meaningful" description

#### Applications:
- Plagiarism detection without language knowledge
- Malware classification by code similarity
- Genome sequence comparison
- Time series anomaly detection

### 2. Grammar-Based Compression (`grammar.hpp`)

**Fundamental Insight**: Data can be represented as formal languages with context-free grammars, achieving compression by extracting hierarchical structure.

#### Algorithms:
- **Sequitur**: Builds hierarchical grammar with no repeated digrams
- **Re-Pair**: Recursive pairing for optimal grammar extraction
- **Straight-Line Programs (SLP)**: Enables O(log n) random access in compressed data
- **Grammar Transform**: LZ77-style compression using grammar rules
- **Smallest Grammar Problem**: Approximation algorithms for minimal grammars

#### Applications:
- XML/JSON compression preserving structure
- Source code compression maintaining syntax
- Compressed data structures with random access
- Pattern mining in sequences

### 3. Adaptive and Online Compression (`adaptive.hpp`)

**Fundamental Insight**: Compression algorithms that learn and adapt during processing can exploit local patterns and temporal locality.

#### Algorithms:
- **Dynamic Huffman Coding**: Updates tree structure in real-time
- **Adaptive Arithmetic Coding**: Online probability model updates
- **Move-to-Front with Local Context**: Exploits both recency and context
- **Recency Rank Encoding**: Weighted combination of recency and frequency
- **Prediction by Partial Matching (PPM)**: Adaptive order context modeling

#### Applications:
- Network packet compression
- Real-time sensor data compression
- Adaptive video streaming
- Online log compression

### 4. Specialized Domain Compressors (`specialized.hpp`)

**Fundamental Insight**: Domain-specific knowledge enables compression ratios far exceeding general-purpose algorithms.

#### Time Series Compression:
- **Gorilla**: Facebook's algorithm for floating-point metrics (XOR + delta-of-delta)
- **Sprintz**: Fire-and-forget prediction for integer sensors

#### Columnar Database Compression:
- **Adaptive encoding selection**: RLE, Dictionary, Bit-packing, Delta, Frame-of-reference
- **Automatic cardinality detection**
- **Sorted/monotonic optimization**

#### Graph Compression:
- **WebGraph**: Copy-list compression exploiting locality
- **K²-tree**: Quadtree representation for sparse adjacency matrices
- **Graph grammars**: Structural pattern extraction

#### Applications:
- Time series databases (InfluxDB-style)
- Columnar analytics (Parquet-style)
- Social network storage
- Genomic sequence databases

### 5. Homomorphic Compression (`homomorphic.hpp`)

**Fundamental Insight**: Compression schemes can preserve certain operations, enabling computation on compressed data without decompression.

#### Algorithms:
- **Searchable Compression**: Pattern matching without decompression
- **Order-Preserving Compression**: Compare and range query compressed data
- **Additive Homomorphic Compression**: Add compressed integers
- **Edit Distance Preserving**: Approximate string similarity on compressed data
- **Format-Preserving Compression**: Maintains data format for legacy systems

#### Applications:
- Encrypted cloud computation
- Compressed database indexes
- Privacy-preserving analytics
- Secure multi-party computation

## Mathematical Foundations

### Information Theory
- **Shannon Entropy**: Theoretical compression limit
- **Kolmogorov Complexity**: Algorithmic information content
- **Rate-Distortion Theory**: Lossy compression bounds

### Formal Language Theory
- **Context-Free Grammars**: Hierarchical pattern representation
- **Straight-Line Programs**: Compressed computation
- **Grammar-Based Complexity**: Smallest grammar problem

### Algebraic Structures
- **Homomorphisms**: Structure-preserving maps
- **Group Actions**: Symmetry in compression
- **Lattice Theory**: Partial order preservation

## Design Philosophy

1. **Compression as Understanding**: Better compression implies better pattern recognition
2. **Compression as Prediction**: All compression is essentially prediction
3. **Compression as Computation**: Programs are compressed data representations
4. **Universal Principles**: Algorithms work without domain-specific knowledge

## Usage Examples

### Computing Similarity Without Domain Knowledge
```cpp
using namespace stepanov::compression::kolmogorov;

normalized_compression_distance<MyCompressor> ncd;
double similarity = ncd.compute(data1, data2);
// Works for ANY data type - text, images, DNA, executables
```

### Random Access in Compressed Data
```cpp
using namespace stepanov::compression::grammar;

slp_compressor<char> slp;
auto grammar = slp.compress(large_file);
char c = slp.access(1000000);  // O(log n) access without decompression!
```

### Computing on Encrypted/Compressed Data
```cpp
using namespace stepanov::compression::homomorphic;

additive_homomorphic_compressor ahc;
auto encrypted_sum = ahc.add_compressed(compressed_a, compressed_b);
// Result is still compressed/encrypted!
```

## Performance Characteristics

| Algorithm | Compression Ratio | Speed | Random Access | Searchable | Homomorphic |
|-----------|------------------|--------|---------------|------------|-------------|
| LZ77 Searchable | Good | Fast | No | Yes | No |
| Grammar-based | Excellent | Medium | O(log n) | Partial | No |
| Adaptive PPM | Excellent | Slow | No | No | No |
| Gorilla (time series) | 12:1 typical | Very Fast | No | No | No |
| Order-preserving | Fair | Fast | Yes | Yes | Yes |
| Homomorphic | Fair | Medium | Yes | Yes | Yes |

## Theoretical Guarantees

- **NCD Metric**: Satisfies metric properties (with normalization)
- **Grammar Size**: O(n/log n) approximation to smallest grammar
- **SLP Access**: O(log n) time, O(n) space
- **Order Preservation**: Exact preservation of total order

## Future Directions

1. **Neural Compression**: Learning compression functions with deep networks
2. **Quantum Compression**: Exploiting quantum superposition
3. **DNA Storage**: Compression for synthetic DNA sequences
4. **Learned Indexes**: ML models as compressed data structures

## References

1. Kolmogorov, A. N. (1965). "Three approaches to the quantitative definition of information"
2. Sequitur: Nevill-Manning & Witten (1997). "Identifying Hierarchical Structure in Sequences"
3. Re-Pair: Larsson & Moffat (2000). "Off-line dictionary-based compression"
4. Gorilla: Pelkonen et al. (2015). "Gorilla: A Fast, Scalable, In-Memory Time Series Database"
5. WebGraph: Boldi & Vigna (2004). "The WebGraph Framework: Compression Techniques"
6. NCD: Cilibrasi & Vitányi (2005). "Clustering by compression"

## Contributing

This module demonstrates that compression is not merely a utility but a fundamental operation revealing deep connections across computer science. Contributions that explore these connections are especially welcome.