# New Generic Math Algorithms & Data Structures

This document summarizes the new algorithms and data structures added to the generic_math library, inspired by the CBT (Computational Basis Transforms) library and following Alex Stepanov's principles of generic programming.

## Overview

The additions focus on mathematical elegance, efficiency through abstraction, and composability. Each implementation:
- Works over abstract mathematical structures (rings, fields, Euclidean domains)
- Provides clear documentation of trade-offs
- Follows the principle of minimal requirements, maximal functionality
- Demonstrates the power of generic programming

## New Components

### 1. Continued Fractions (`continued_fraction.hpp`)

**Purpose**: Rational approximation and number theory algorithms

**Key Features**:
- Convert between rationals and continued fraction representation
- Compute convergents (best rational approximations)
- Solve linear Diophantine equations via extended GCD
- Pell equation solver
- Works over any Euclidean domain

**Example Use Cases**:
- Finding best rational approximations with bounded denominators
- Solving ax + by = c for integers x, y
- Computing fundamental solutions to Pell equations

### 2. Binary Indexed Tree / Fenwick Tree (`fenwick_tree.hpp`)

**Purpose**: Efficient range queries and updates

**Key Features**:
- O(log n) range sum queries
- O(log n) point updates
- Generic over any invertible monoid (not just addition)
- 2D variant for matrix queries
- Range update variant with lazy propagation
- Order statistics tree implementation

**Trade-offs vs Segment Tree**:
- More memory efficient (n vs 4n)
- Simpler implementation
- Requires invertible operation
- Cannot handle arbitrary range updates as efficiently

### 3. p-adic Numbers (`padic.hpp`)

**Purpose**: Alternative completion of rationals based on divisibility

**Key Features**:
- Arithmetic in p-adic fields
- Hensel lifting for equation solving
- p-adic valuation and norms
- Natural framework for modular arithmetic

**Applications**:
- Number theory computations
- Solving polynomial equations modulo prime powers
- Understanding local properties of numbers

### 4. Combinatorial Algorithms (`combinatorics.hpp`)

**Purpose**: Generic implementations of combinatorial functions

**Key Features**:
- Pascal's triangle (binomial coefficients)
- Stirling numbers (both kinds)
- Integer partitions with generation
- Catalan numbers
- Bell numbers
- Derangements
- Eulerian numbers
- Works over any ring (not just integers)

**Design Philosophy**:
- Pre-computation for repeated queries
- Both recursive and explicit formulas
- Generation of combinatorial objects

### 5. Sparse Matrix (`sparse_matrix.hpp`)

**Purpose**: Efficient operations on mostly-zero matrices

**Key Features**:
- Coordinate (COO) format storage
- Conversion to CSR format
- All standard matrix operations
- Kronecker product
- Block matrix construction
- Works over any ring

**Trade-offs**:
- Space: O(nnz) vs O(mn) for dense
- Multiplication: Better for sparse matrices
- Element access: O(log nnz) vs O(1)

### 6. Fast Fourier Transform for Rings (`fft.hpp`)

**Purpose**: Polynomial multiplication and convolution in various domains

**Key Features**:
- Classical complex FFT
- Number Theoretic Transform (NTT) for modular arithmetic
- Generic DFT for any ring with roots of unity
- 2D FFT for image processing
- Bluestein's algorithm for arbitrary sizes

**Applications**:
- Fast polynomial multiplication
- Signal processing
- Large integer multiplication
- Convolution in various algebraic structures

### 7. Primality Testing (`primality.hpp`)

**Purpose**: Comprehensive primality testing algorithms

**Key Features**:
- Miller-Rabin (probabilistic, fast)
- Solovay-Strassen (uses Jacobi symbol)
- Baillie-PSW (very strong test)
- Lucas-Lehmer for Mersenne primes
- Trial division
- Sieve of Eratosthenes
- Segmented sieve for large ranges
- Pollard's rho factorization

**Design**:
- Generic over integral domains
- Both deterministic and probabilistic variants
- Optimized for different number ranges

## Design Principles Demonstrated

### 1. Generic Programming
All algorithms are template-based and work with types satisfying mathematical concepts rather than concrete types. For example, continued fractions work over any Euclidean domain, not just integers.

### 2. Efficiency Through Abstraction
By working at the right level of abstraction, we achieve both generality and efficiency. The Fenwick tree, for instance, works for any group operation while maintaining O(log n) complexity.

### 3. Explicit Trade-offs
Each data structure clearly documents its trade-offs. Sparse matrices trade O(1) access for space efficiency; p-adic numbers provide a different notion of "closeness" than real numbers.

### 4. Composability
Components are designed to work together. For example:
- Sparse matrices can store polynomial coefficients
- FFT can work over modular arithmetic rings
- Combinatorial algorithms can compute over polynomial rings

### 5. Mathematical Rigor
Implementations follow mathematical definitions closely while remaining practical. Concepts like rings, fields, and Euclidean domains are properly modeled.

## Integration with Existing Library

These new components integrate seamlessly with the existing generic_math library:
- Use the same concept system (`euclidean_domain`, `ring`, etc.)
- Follow the same naming conventions
- Build on existing primitives (`quotient`, `remainder`, `norm`)
- Compatible with existing adaptors for built-in types

## Inspired by CBT Library

The CBT (Computational Basis Transforms) library inspired several design decisions:
- **Transform mindset**: View algorithms as transformations between computational domains
- **Trade-off awareness**: Every representation has advantages and costs
- **Composition power**: Combining transforms yields multiplicative benefits
- **Domain expertise**: Choose the right representation for the problem

## Usage Example

```cpp
#include <generic_math/continued_fraction.hpp>
#include <generic_math/builtin_adaptors.hpp>

using namespace generic_math;

// Find best rational approximation to π
auto cf = to_continued_fraction(355, 113);
auto convergents = compute_convergents(cf);

// Solve Diophantine equation
auto solution = solve_linear_diophantine(15, 21, 9);
if (solution.has_solution) {
    // x = solution.x0 + solution.dx * k
    // y = solution.y0 - solution.dy * k
}
```

## Future Directions

Potential additions following the same principles:
- Elliptic curves for cryptography
- Lattice reduction algorithms
- Tensor operations with Einstein notation
- Automatic differentiation
- Interval arithmetic
- Tropical algebra
- Persistent data structures

## Conclusion

These additions demonstrate that generic programming principles lead to:
- More reusable code (works across different mathematical structures)
- Better performance (through proper abstractions)
- Clearer interfaces (concepts make requirements explicit)
- Deeper understanding (implementations follow mathematical definitions)

The library now provides a rich set of mathematical algorithms that are both theoretically sound and practically efficient, following Alex Stepanov's vision of generic programming.