---
author:
  name: "Alex Towell"
  email: "queelius@gmail.com"
  url: "https://metafunctor.com"
title: "Is It Prime?"
date: 2019-09-10
draft: false
tags:
  - C++
  - algorithms
  - number-theory
  - primality
  - probabilistic-algorithms
categories:
  - Computer Science
  - Mathematics
series: ["stepanov"]
series_weight: 3
math: true
description: "The Miller-Rabin primality test demonstrates how probabilistic algorithms can achieve arbitrary certainty, trading absolute truth for practical efficiency."
---

*The Miller-Rabin primality test and the mathematics of certainty*

## The Problem

Given a large number n, is it prime? Trial division up to sqrt(n) is too slow for cryptographic-sized numbers. We need something faster---and we're willing to accept "probably prime" with quantifiable certainty.

## Fermat's Little Theorem

For prime \(p\) and any \(a\) not divisible by \(p\):

$$a^{p-1} \equiv 1 \pmod{p}$$

This suggests a test: pick random a, compute a^(n-1) mod n. If the result isn't 1, n is definitely composite. But if it *is* 1, n might be prime... or might be a Carmichael number that fools this test.

## The Miller-Rabin Improvement

Miller and Rabin observed something stronger. For odd prime \(p\), write \(p-1 = 2^r \cdot d\) (factor out all 2s). Then the sequence:

$$a^d, a^{2d}, a^{4d}, \ldots, a^{2^r \cdot d} = a^{p-1}$$

must either:
1. Start with 1, or
2. Contain -1 (i.e., p-1) somewhere before reaching 1

Why? Because the only square roots of 1 mod p are plus or minus 1. If we ever see 1 without first seeing -1, we've found a non-trivial square root of 1, proving n is composite.

## The Witness Test

```cpp
bool witness_test(int64_t n, int64_t a) {
    // Write n-1 = 2^r x d
    int64_t d = n - 1;
    int r = 0;
    while ((d & 1) == 0) { d >>= 1; r++; }

    // Compute x = a^d mod n
    int64_t x = mod_pow(a, d, n);

    if (x == 1 || x == n - 1) return true;  // Probably prime

    // Square r-1 times, looking for n-1
    for (int i = 1; i < r; i++) {
        x = (x * x) % n;
        if (x == n - 1) return true;
    }

    return false;  // Definitely composite
}
```

If `witness_test(n, a)` returns false, n is **definitely composite**---a is a "witness" to compositeness.

## Error Bounds

The beautiful part: for any composite n, **at least 3/4 of all possible witnesses a in [2, n-2] will detect it**. This means each random witness has at most 1/4 chance of failing to detect a composite.

With \(k\) independent witnesses:

$$P(\text{false positive}) \leq \left(\frac{1}{4}\right)^k$$

| Witnesses | Error bound |
|-----------|-------------|
| 10 | \(< 10^{-6}\) |
| 20 | \(< 10^{-12}\) |
| 40 | \(< 10^{-24}\) |

## Parameterizing by Error

Rather than asking "how many iterations?", ask "what error rate is acceptable?":

```cpp
bool is_prime(int64_t n, double max_error = 1e-12) {
    int k = ceil(-log(max_error) / log(4));
    // ... test with k random witnesses
}
```

The implementation also returns the actual error bound achieved:

```cpp
auto result = is_prime_with_error(n);
if (result.probably_prime) {
    std::cout << "Prime with error < " << result.error_bound << "\n";
}
```

## Modular Exponentiation

The workhorse is computing a^e mod n efficiently. The peasant algorithm again:

```cpp
int64_t mod_pow(int64_t base, int64_t exp, int64_t m) {
    int64_t result = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1) result = (result * base) % m;
        base = (base * base) % m;
        exp >>= 1;
    }
    return result;
}
```

O(log e) multiplications, each followed by a mod operation.

## Carmichael Numbers

The Fermat test fails on Carmichael numbers---composites where \(a^{n-1} \equiv 1 \pmod{n}\) for all \(a\) coprime to \(n\). The smallest is \(561 = 3 \times 11 \times 17\).

Miller-Rabin correctly identifies these as composite because it checks the *path* to a^(n-1), not just the final value.

```cpp
EXPECT_FALSE(is_prime(561));   // Fermat test would fail here
EXPECT_FALSE(is_prime(1729));  // The Hardy-Ramanujan number
```

## Why Not Deterministic?

There exist deterministic witnesses that work for all \(n\) below certain bounds. With the first 12 primes as witnesses, Miller-Rabin is deterministic for all \(n < 3.3 \times 10^{24}\).

But the probabilistic version is:
1. Simpler to understand
2. Sufficient for practice (10^-12 error is smaller than hardware error rates)
3. More pedagogically honest about what we're computing

## Further Reading

- Miller, "Riemann's Hypothesis and Tests for Primality" (1976)
- Rabin, "Probabilistic Algorithm for Testing Primality" (1980)
