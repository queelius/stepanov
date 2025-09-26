# Generic Math Numeric Types

This document describes the three new numeric type classes added to the generic_math library, following Alex Stepanov's principles of generic programming.

## Overview

The library now provides three fundamental numeric types that integrate seamlessly with all existing generic algorithms:

1. **`bounded_integer<N>`** - Fixed-size arbitrary precision signed integers
2. **`rational<T>`** - Exact rational numbers with support for infinities
3. **`fixed_decimal<T, Scale>`** - Efficient fixed-point decimal arithmetic

All types are designed to:
- Model appropriate mathematical concepts (field, euclidean_domain, etc.)
- Work with existing generic algorithms (gcd, power, sum, product)
- Support the fundamental operations (twice, half, even, increment, decrement)
- Integrate with the type erasure system
- Maintain mathematical correctness and elegance

## bounded_integer<N>

A fixed-size arbitrary precision signed integer type using two's complement representation.

### Features
- Template parameter `N` specifies the number of bytes
- Two's complement representation for negative numbers
- Supports all integer operations
- Provides `abs()` method returning `bounded_nat<N>`
- Models `euclidean_domain` concept

### Example Usage
```cpp
using int128 = bounded_integer<16>;  // 128-bit signed integer

int128 a(42);
int128 b(-17);
int128 c = a + b;  // 25
int128 pow = power(int128(2), int128(10));  // 1024

// Works with GCD algorithm
int128 x(48), y(18);
int128 g = gcd(x, y);  // 6
```

### Design Decisions
- Uses two's complement for consistency with built-in types
- Sign determined from most significant bit
- Little-endian byte ordering for efficiency
- Conversion to/from built-in integral types

## rational<T>

Exact rational number representation as numerator/denominator pairs.

### Features
- Works with any integral type or type modeling `integral_domain`
- Always maintains reduced form (coprime numerator/denominator)
- Supports infinities (denominator = 0)
- Supports epsilon (smallest positive rational)
- Integration with continued fractions
- Models `field` concept when finite

### Special Values
- **Zero**: 0/1
- **Infinity**: ±1/0
- **Epsilon**: 1/max_value

### Example Usage
```cpp
using rat = rational<int>;

rat a(6, 8);   // Automatically reduced to 3/4
rat b(2, 3);

rat sum = a + b;        // 17/12
rat prod = a * b;      // 1/2
rat quot = a / b;      // 9/8

// Continued fraction support
auto cf = a.to_continued_fraction();  // [0, 1, 3]

// Best approximation with bounded denominator
rat pi(355, 113);
rat approx = best_approximation(pi, 100);  // 22/7

// Special operations
rat m = mediant(rat(1, 2), rat(2, 3));  // 3/5
```

### Mathematical Properties
- Forms a field (when excluding infinities)
- Totally ordered
- Supports floor, ceiling, fractional part operations
- Mediant operation for Farey sequences

## fixed_decimal<T, Scale>

Fixed-point decimal arithmetic with compile-time precision.

### Features
- Template parameter `Scale` specifies decimal places
- Internally stores as scaled integer (value × 10^Scale)
- Exact decimal arithmetic (no binary rounding errors)
- Efficient operations using integer arithmetic
- Models `field` concept

### Trade-offs vs Floating Point

**Advantages:**
- Exact decimal arithmetic
- Predictable precision and range
- Faster on systems without FPU
- Deterministic across platforms

**Disadvantages:**
- Limited range compared to floating point
- Fixed precision (can't adapt to magnitude)
- More expensive division and transcendental functions

### Example Usage
```cpp
using money = fixed_decimal<long long, 2>;  // Currency with 2 decimal places

money price(19.99);
money tax_rate(0.08);
money tax = price * tax_rate;  // 1.60

// Compound interest calculation
money rate(1.05);  // 5% growth
money result = power(rate, 10);  // 1.61

// Scale conversion
using precise = fixed_decimal<long long, 6>;
precise high_precision(3.141593);
money rounded = scale_convert<long long, 6, 2>(high_precision);  // 3.14

// Square root approximation
money value(2.0);
money sqrt_val = sqrt(value);  // 1.41
```

### Rounding Behavior
- Multiplication and division round to nearest
- Explicit rounding functions: floor, ceil, round, trunc
- Scale conversion performs rounding

## Integration with Generic Algorithms

All three types seamlessly integrate with existing generic algorithms:

### GCD Algorithm
```cpp
bounded_integer<8> bi_a(15), bi_b(10);
auto bi_gcd = gcd(bi_a, bi_b);  // 5

int r_gcd = gcd(rat(15, 1).numerator(), rat(10, 1).numerator());  // 5
```

### Power Algorithm
```cpp
bounded_integer<8> base(2), exp(8);
auto result = power(base, exp);  // 256

fixed_decimal<long, 4> fd_base(2.0);
auto fd_result = power(fd_base, 8);  // 256.0000
```

### Sum/Product
All types work with the generic sum and product algorithms through their arithmetic operators.

## Fundamental Operations

All types provide the required fundamental operations for generic algorithms:

- `even(x)` - Check if even
- `twice(x)` - Multiply by 2
- `half(x)` - Divide by 2
- `increment(x)` - Add 1
- `decrement(x)` - Subtract 1

These operations enable the use of efficient generic algorithms like Russian peasant multiplication.

## Mathematical Concepts

The types model the following concepts:

- **bounded_integer<N>**: `euclidean_domain`, `ordered_ring`
- **rational<T>**: `field` (when finite), `ordered_field`
- **fixed_decimal<T,S>**: `field`, `ordered_field`

## Performance Considerations

1. **bounded_integer**: O(N) space, O(N) addition, O(N²) multiplication
2. **rational**: Reduction cost on each operation, continued fraction conversion is O(log n)
3. **fixed_decimal**: Same as underlying integer type, very efficient

## Future Enhancements

Potential improvements:
- Optimized multiplication algorithms for large bounded_integer
- Lazy evaluation for rational arithmetic
- SIMD operations for fixed_decimal
- Transcendental functions for all types
- Serialization support

## Conclusion

These three numeric types demonstrate the power of generic programming:
- Mathematical correctness through proper abstraction
- Composability with existing algorithms
- Efficiency through compile-time abstraction
- Clear semantics for edge cases

They follow Alex Stepanov's principle: "The library should provide a small number of highly reusable components that can be combined in many ways."