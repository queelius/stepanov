# Stepanov

A modern C++20/23 header-only library for generic mathematical algorithms inspired by Alex Stepanov's principles and the C++ standard library design. This library demonstrates the power of generic programming through mathematical abstractions and efficient implementations using C++20 concepts.

## Philosophy

This library embodies three core programming philosophies:
- **Unix Philosophy**: "Do one thing well" - Each component has a single, clear responsibility
- **Pythonic Philosophy**: "There should be one obvious way to do it" - APIs are intuitive and consistent
- **C++ Generic Programming**: "Don't pay for what you don't use" - Zero-cost abstractions through templates

## Features

### Mathematical Concepts
- **Algebraic Structures**: Groups, Rings, Fields, Euclidean Domains
- **Generic Algorithms**: Work with any type satisfying required concepts
- **Composable Operations**: Build complex algorithms from simple primitives

### Core Components

#### 1. **Concepts** (`concepts.hpp`)
Modern C++20 concepts defining mathematical abstractions:
```cpp
template<typename T>
concept euclidean_domain = integral_domain<T> && requires(T a, T b) {
    { quotient(a, b) } -> std::convertible_to<T>;
    { remainder(a, b) } -> std::convertible_to<T>;
    { norm(a) } -> std::integral;
};
```

#### 2. **Generic Algorithms** (`algorithms.hpp`)
Elegant algorithms following Stepanov's principles:
- Orbit and cycle detection (Floyd's & Brent's algorithms)
- Generic accumulation with early termination
- Three-way partitioning
- Function composition
- K-way merge

#### 3. **Number Theory** (`gcd.hpp`)
- GCD for any Euclidean domain
- Extended GCD with Bézout coefficients
- Binary GCD (Stein's algorithm)
- Chinese Remainder Theorem solver
- Modular inverse

#### 4. **Polynomials** (`polynomial.hpp`)
- Sparse polynomial representation
- Arithmetic operations
- Root finding (Newton-Raphson)
- Calculus operations (derivative, antiderivative)

#### 5. **Arbitrary Precision** (`bounded_nat.hpp`)
- Fixed-size arbitrary precision natural numbers
- Demonstrates generic programming with custom types
- Efficient bit operations

#### 6. **Type Erasure** (`type_erasure.hpp`)
- Runtime polymorphism with value semantics
- Following Sean Parent's techniques
- Algebraic type erasure
- Iterator type erasure

## Usage Examples

### Basic Mathematical Operations
```cpp
#include <stepanov/math.hpp>
#include <stepanov/builtin_adaptors.hpp>

using namespace stepanov;

// Works with built-in types
auto result = power_accumulate(2, 10, 1);  // 2^10 = 1024

// Works with custom types that provide required operations
bounded_nat<8> a(100), b(200);
auto sum = a + b;  // Uses generic sum algorithm
```

### GCD and Number Theory
```cpp
#include <stepanov/gcd.hpp>

// Binary GCD for integers
auto g = binary_gcd(48, 18);  // Returns 6

// Extended GCD
auto [gcd, x, y] = extended_gcd(30, 18);
// Returns gcd=6, with 30*x + 18*y = 6

// Works with any Euclidean domain
polynomial<double> p1{{2, 1.0}}, p2{{1, 1.0}};
auto poly_gcd = gcd(p1, p2);
```

### Polynomial Operations
```cpp
#include <stepanov/polynomial.hpp>

// Create polynomial: x^2 + 2x + 1
polynomial<double> p{{0, 1.0}, {1, 2.0}, {2, 1.0}};

// Evaluate at x = 3
double value = p(3.0);  // Returns 16.0

// Find roots
newton_solver<double> solver;
auto root = solver.find_root(p, 1.0);

// Calculus
auto dp = derivative(p);  // 2x + 2
```

### Generic Algorithms
```cpp
#include <stepanov/algorithms.hpp>

// Detect cycles in transformations
auto collatz = [](int n) {
    return (n % 2 == 0) ? n / 2 : 3 * n + 1;
};
auto cycle = detect_cycle(27, collatz);

// Function composition
auto f = compose(
    [](int x) { return x + 1; },
    [](int x) { return x * 2; }
);  // f(x) = 2x + 1

// Three-way partition
std::vector<int> data{3, 1, 4, 1, 5, 9, 2, 6};
auto [less, equal] = partition_3way(data.begin(), data.end(), 4);
```

### Type Erasure
```cpp
#include <stepanov/type_erasure.hpp>

// Store different types with same interface
any_algebraic a(5), b(3.14);
auto sum = a + b;  // Runtime polymorphism

// Function type erasure
any_function<int(int)> func = [](int x) { return x * x; };
int result = func(5);  // Returns 25
```

## Building and Testing

This is a header-only library - no build required!

### Requirements
- C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- Concepts support enabled

### Running Tests
```bash
# Compile tests
g++ -std=c++20 -I include -O3 test/test_stepanov.cpp -o test_stepanov

# Run tests
./test_stepanov
```

### Using in Your Project
Simply copy the `include/stepanov` directory to your project and include the headers you need:

```cpp
#include <stepanov/concepts.hpp>
#include <stepanov/algorithms.hpp>
// ... use the library
```

## Design Principles

1. **Generic Programming First**: All algorithms work with any type satisfying minimal requirements
2. **Concepts-Based Design**: Clear, checkable requirements for template parameters
3. **Efficiency Through Abstraction**: Use fundamental operations as building blocks
4. **Composability**: Small, focused components that combine elegantly
5. **Mathematical Correctness**: Algorithms based on proven mathematical principles

## Files Overview

- `include/stepanov/concepts.hpp` - C++20 concepts for mathematical abstractions
- `include/stepanov/math.hpp` - Basic mathematical operations using generic programming
- `include/stepanov/algorithms.hpp` - Advanced generic algorithms (Stepanov-style)
- `include/stepanov/gcd.hpp` - GCD and number theory algorithms for Euclidean domains
- `include/stepanov/polynomial.hpp` - Sparse polynomial implementation with Newton's method
- `include/stepanov/bounded_nat.hpp` - Fixed-size arbitrary precision natural numbers
- `include/stepanov/type_erasure.hpp` - Runtime polymorphism with value semantics
- `include/stepanov/builtin_adaptors.hpp` - Adaptors for built-in types
- `test/test_stepanov.cpp` - Comprehensive test suite

## Inspiration

This library is inspired by:
- **Alex Stepanov's** work on generic programming and "Elements of Programming"
- **The STL** design principles and iterator concepts
- **SICP** emphasis on abstraction and composition
- **Rich Hickey's** ideas on simplicity
- **Category Theory** insights into composition

## License

MIT License - See LICENSE file for details