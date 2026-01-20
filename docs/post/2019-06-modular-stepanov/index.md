---
title: "Modular Arithmetic as Rings"
date: 2019-06-22
draft: false
tags:
  - C++
  - generic-programming
  - number-theory
  - rings
  - algebra
categories:
  - Computer Science
  - Mathematics
series: ["stepanov"]
series_weight: 2
math: true
description: "Integers modulo N form a ring---an algebraic structure that determines which algorithms apply. Understanding this structure unlocks algorithms from cryptography to competitive programming."
---

*Finite algebraic structures and what they teach us about algorithms*

## The Stepanov Perspective

Alex Stepanov's central insight: **algorithms arise from algebraic structure**. The same algorithm that works on integers works on matrices, polynomials, and modular integers---not by accident, but because they share algebraic properties.

Integers modulo N form a **ring**: a set with addition and multiplication satisfying familiar laws. When N is prime, it's a **field**---every non-zero element has a multiplicative inverse. Understanding these structures teaches us which algorithms apply where.

## The Ring Z/NZ

Integers modulo N, written Z/NZ, are equivalence classes:
- 0 = {..., -N, 0, N, 2N, ...}
- 1 = {..., -N+1, 1, N+1, 2N+1, ...}
- ...

Operations are inherited from integers:
```
[a] + [b] = [a + b]
[a] x [b] = [a x b]
```

The implementation keeps one representative per class, in [0, N):

```cpp
template<int64_t N>
struct mod_int {
    int64_t v;  // Always in [0, N)

    static constexpr int64_t normalize(int64_t x) {
        x %= N;
        return x < 0 ? x + N : x;
    }

    constexpr mod_int(int64_t x) : v(normalize(x)) {}
};
```

## Ring Axioms

A ring (R, +, x) satisfies:

**Addition forms an abelian group:**
- Associative: (a + b) + c = a + (b + c)
- Identity: a + 0 = a
- Inverses: a + (-a) = 0
- Commutative: a + b = b + a

**Multiplication is a monoid:**
- Associative: (a x b) x c = a x (b x c)
- Identity: a x 1 = 1 x a = a

**Distributive:**
- a x (b + c) = a x b + a x c
- (b + c) x a = b x a + c x a

These axioms *enable algorithms*. Power-by-squaring works because multiplication is associative. The extended GCD works because of the ring structure.

## Fermat's Little Theorem

When N is prime, something special happens: every non-zero element has a multiplicative inverse. The set of non-zero elements forms a multiplicative group of order N-1.

Fermat's Little Theorem: for prime \(p\) and \(a\) not congruent to \(0 \pmod{p}\):

$$a^{p-1} \equiv 1 \pmod{p}$$

This gives us the inverse:

$$a \cdot a^{p-2} = a^{p-1} \equiv 1 \pmod{p}$$

So \(a^{p-2}\) is the multiplicative inverse of \(a\):

```cpp
constexpr mod_int inverse() const {
    return pow(N - 2);  // Using repeated squaring
}
```

## The Connection to Peasant

The power function uses the same peasant algorithm from elsewhere:

```cpp
constexpr mod_int pow(int64_t exp) const {
    mod_int result(1), base = *this;
    while (exp > 0) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}
```

This works because:
1. `mod_int` multiplication is associative
2. `mod_int(1)` is the multiplicative identity

Same algorithm, different type. The structure (monoid) determines applicability.

## Algebraic Operations for Generic Algorithms

To use `mod_int` with generic algorithms, we provide the required operations:

```cpp
template<int64_t N>
constexpr mod_int<N> zero(mod_int<N>) { return mod_int<N>(0); }

template<int64_t N>
constexpr mod_int<N> one(mod_int<N>) { return mod_int<N>(1); }

template<int64_t N>
constexpr mod_int<N> twice(mod_int<N> x) { return x + x; }

template<int64_t N>
constexpr mod_int<N> half(mod_int<N> x) {
    return x * mod_int<N>(2).inverse();
}
```

Now generic `power()` and `product()` work with modular integers.

## Why Compile-Time Modulus?

The modulus N is a template parameter:

```cpp
mod_int<7> a;       // Mod 7
mod_int<1000000007> b;  // Mod 10^9+7
```

Benefits:
- No runtime storage for modulus
- Compiler can optimize division/modulo
- Type safety: can't accidentally mix mod-7 with mod-11

Trade-off: can't choose modulus at runtime. For that, you'd need a runtime-parameterized version.

## Common Moduli

Competitive programming conventions:
- **10^9 + 7** --- Large prime, fits in 32 bits when squared
- **10^9 + 9** --- Another large prime
- **998244353** --- 2^23 x 7 x 17 + 1, useful for NTT

```cpp
using mod_1e9_7 = mod_int<1000000007>;
```

## Connections

Modular arithmetic appears throughout:

1. **Cryptography**: RSA uses modular exponentiation
2. **Hash functions**: Often compute mod some prime
3. **Error detection**: CRC and checksums
4. **Miller-Rabin**: Primality testing
5. **Chinese Remainder Theorem**: Solving systems of congruences

## The Deeper Point

Stepanov teaches us to see *structure first, algorithm second*. When you recognize "this is a ring," you immediately know which algorithms apply.

Modular arithmetic isn't just "integers with %". It's a complete algebraic structure with its own theorems and algorithms. Understanding the structure unlocks the algorithms.

## Further Reading

- Stepanov & McJones, *Elements of Programming*, Chapter 5
- Ireland & Rosen, *A Classical Introduction to Modern Number Theory*
