# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A header-only C++ library for generic mathematical algorithms inspired by Alex Stepanov's principles and the C++ standard library design. The library emphasizes generic programming, mathematical abstractions, and efficient implementations using concepts lite (C++20 concepts).

## Architecture & Design Philosophy

### Core Principles
- **Generic Programming**: All algorithms are template-based and work with any type satisfying required concepts
- **Algebraic Structures**: Functions operate on mathematical abstractions (groups, rings, fields, Euclidean domains)
- **Efficiency Through Abstraction**: Use fundamental operations (half, twice, increment) as building blocks
- **Concepts-Based Design**: Types must model specific mathematical structures with well-defined axioms

### Key Components

1. **Math Operations** (`math.hpp`):
   - Generic implementations of `product`, `sum`, `power`, `square`
   - Built on primitive operations: `twice`, `half`, `even`, `increment`, `decrement`
   - Types must provide these operations to use the generic algorithms

2. **Number Theory** (`gcd.hpp`, `mod.hpp`, `expmod.hpp`):
   - GCD for Euclidean domains (requires `quotient`, `remainder`, `norm`)
   - Modular arithmetic and exponentiation
   - Fermat primality testing

3. **Data Structures**:
   - `bounded_nat<N>`: Fixed-size arbitrary precision natural numbers
   - `polynomial<T>`: Sparse polynomial representation with Newton's method root finding

4. **Group Theory** (`groups/`):
   - Abstract group operations and axiom validation
   - Finite group enumeration and testing
   - Abelian group operations

## Development Commands

### Building and Testing
```bash
# This is a header-only library - no build needed for the library itself
# To test compilation of headers:
g++ -std=c++20 -I include -fsyntax-only include/generic_math/*.hpp

# For testing individual components (create test files):
g++ -std=c++20 -I include -O3 test_program.cpp -o test_program

# Enable concepts checking:
g++ -std=c++20 -fconcepts-diagnostics-depth=2 -I include test.cpp
```

### Code Style Guidelines
- Use snake_case for all identifiers
- Template parameters use PascalCase (e.g., `EuclideanDomain`)
- Concepts should clearly state mathematical requirements in comments
- Prefer free functions over member functions for generic algorithms

## Refactoring Priorities for Generic Programming

1. **Add C++20 Concepts**: Replace commented-out concept definitions with proper C++20 concepts
2. **Fix Compilation Errors**: Several files have syntax errors (e.g., `polynomial.hpp` line 199: `polynormial` typo)
3. **Standardize Operations**: Ensure all types provide required operations as free functions
4. **Complete Group Theory Implementation**: `.cpp` files in `groups/` should become header-only templates
5. **Add Iterator Support**: Types should provide STL-compatible iterators where appropriate

## Testing Strategy
- Create comprehensive test suite using property-based testing for mathematical properties
- Test algebraic axioms (associativity, commutativity, distributivity)
- Verify concepts are properly constrained
- Performance benchmarks comparing to standard implementations

## Example Usage Pattern
```cpp
#include <generic_math/gcd.hpp>

// Type must model EuclideanDomain concept
template<typename T>
requires EuclideanDomain<T>
T extended_gcd(T a, T b, T& x, T& y) {
    // Implementation using generic operations
}
```

## Important Notes
- This library requires C++20 or later for concepts support
- All algorithms should be constexpr where possible
- Focus on mathematical correctness over premature optimization
- Document mathematical requirements and axioms for each concept